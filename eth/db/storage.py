from typing import (
    FrozenSet,
    List,
    NamedTuple,
    Set,
)

from eth_hash.auto import (
    keccak,
)
from eth_typing import (
    Address,
    Hash32,
)
from eth_utils import (
    ValidationError,
    encode_hex,
    get_extended_debug_logger,
    int_to_big_endian,
    to_bytes,
    to_int,
)
import rlp
from trie import (
    HexaryTrie,
    exceptions as trie_exceptions,
)
from web3 import Web3
from web3.types import (
    BlockIdentifier,
)

from eth._utils.padding import (
    pad32,
)
from eth.abc import (
    AccountStorageDatabaseAPI,
    AtomicDatabaseAPI,
    DatabaseAPI,
)
from eth.constants import (
    BLANK_ROOT_HASH,
)
from eth.db.backends.base import (
    BaseDB,
)
from eth.db.backends.memory import (
    MemoryDB,
)
from eth.db.batch import (
    BatchDB,
)
from eth.db.cache import (
    CacheDB,
)
from eth.db.journal import (
    JournalDB,
)
from eth.typing import (
    JournalDBCheckpoint,
)
from eth.vm.interrupt import (
    MissingStorageTrieNode,
)

ZERO_VALUE = b'\x00' * 32

class PendingWrites(NamedTuple):
    """
    A set of variables captured just before account storage deletion.
    The variables are used to revive storage if the EVM reverts to a point
    prior to deletion.
    """

    write_trie: HexaryTrie  # The write trie at the time of deletion
    trie_nodes_batch: BatchDB  # A batch of all trie nodes written to the trie
    starting_root_hash: Hash32  # The starting root hash


class StorageLookup(BaseDB):
    """
    This lookup converts lookups of storage slot integers into the appropriate trie
    lookup. Similarly, it persists changes to the appropriate trie at write time.

    StorageLookup also tracks the state roots changed since the last persist.
    """

    logger = get_extended_debug_logger("eth.db.storage.StorageLookup")

    # The trie that is modified in-place, used to calculate storage root on-demand
    _write_trie: HexaryTrie

    # These are the new trie nodes, waiting to be committed to disk
    _trie_nodes_batch: BatchDB

    # When deleting an account, push the pending write info onto this stack.
    # This stack can get as big as the number of transactions per block: one for
    # each delete.
    _historical_write_tries: List[PendingWrites]

    def __init__(
            self,
            db: DatabaseAPI,
            w3: Web3,
            block_identifier: BlockIdentifier,
            address: Address
        ) -> None:
        self._db = db
        self._w3 = w3
        self._block_identifier = block_identifier
        self._address = address

    def _make_db_key(self, key: bytes) -> bytes:
        return b'storage' + bytes(self._address) + key

    def __getitem__(self, key: bytes) -> bytes:
        k = self._make_db_key(key)

        if k not in self._db:
            val = self._w3.eth.get_storage_at(
                self._address,
                to_int(key),
                block_identifier=self._block_identifier
            )
            self._db[k] = val
        
        return self._db[k]

    def __setitem__(self, key: bytes, value: bytes) -> None:
        k = self._make_db_key(key)
        self._db[k] = value

    def _exists(self, key: bytes) -> bool:
        # used by BaseDB for __contains__ checks
        raise NotImplementedError("StorageLookup does not support __contains__")
        hashed_slot = self._decode_key(key)
        read_trie = self._get_read_trie()
        return hashed_slot in read_trie

    def __delitem__(self, key: bytes) -> None:
        k = self._make_db_key(key)
        self._db[k] = ZERO_VALUE

    def commit_to(self, db: DatabaseAPI) -> None:
        """
        Trying to commit changes when nothing has been written will raise a
        ValidationError
        """
        raise NotImplementedError("StorageLookup does not support commit_to")
        self.logger.debug2("persist storage root to data store")
        if self._trie_nodes_batch is None:
            raise ValidationError(
                "It is invalid to commit an account's storage if it has no pending "
                "changes. Always check storage_lookup.has_changed_root before "
                "attempting to commit. Write tries on stack = "
                f"{len(self._historical_write_tries)}; Root hash = "
                f"{encode_hex(self._starting_root_hash)}"
            )
        self._trie_nodes_batch.commit_to(db, apply_deletes=False)

        # Mark the trie as having been all written out to the database.
        # It removes the 'dirty' flag and clears out any pending writes.
        self._initialize_to_root_hash(self._write_trie.root_hash)

    def new_trie(self) -> int:
        """
        Switch to an empty trie. Save the old trie, and pending writes, in
        case of a revert.

        :return: index for reviving the previous trie
        """
        raise NotImplementedError("StorageLookup does not support new_trie")
        write_trie = self._get_write_trie()

        # Write the previous trie into a historical stack
        self._historical_write_tries.append(
            PendingWrites(
                write_trie,
                self._trie_nodes_batch,
                self._starting_root_hash,
            )
        )

        new_idx = len(self._historical_write_tries)
        self._starting_root_hash = BLANK_ROOT_HASH
        self._write_trie = None
        self._trie_nodes_batch = None

        return new_idx

    def rollback_trie(self, trie_index: int) -> None:
        """
        Revert back to the previous trie, using the index returned by a
        :meth:`~new_trie` call. The index returned by that call returns you
        to the trie in place *before* the call.

        :param trie_index: index for reviving the previous trie
        """

        if trie_index >= len(self._historical_write_tries):
            raise ValidationError(
                f"Trying to roll back a delete to index {trie_index}, but there are "
                f"only {len(self._historical_write_tries)} indices available."
            )

        (
            self._write_trie,
            self._trie_nodes_batch,
            self._starting_root_hash,
        ) = self._historical_write_tries[trie_index]

        # Cannot roll forward after a rollback, so remove created/ignored tries.
        # This also deletes the trie that you just reverted to. It will be re-added
        # to the stack when the next new_trie() is called.
        del self._historical_write_tries[trie_index:]


CLEAR_COUNT_KEY_NAME = b"clear-count"


class AccountStorageDB(AccountStorageDatabaseAPI):
    logger = get_extended_debug_logger("eth.db.storage.AccountStorageDB")

    def __init__(
        self,
        db: AtomicDatabaseAPI,
        w3: Web3,
        block_identifier: BlockIdentifier,
        address: Address
    ) -> None:
        """
        Database entries go through several pipes, like so...

        .. code::

          db -> _storage_lookup -> _storage_cache -> _locked_changes -> _journal_storage

        db is the raw database, we can assume it hits disk when written to.
        Keys are stored as node hashes and rlp-encoded node values.

        _storage_lookup is itself a pair of databases: (BatchDB -> HexaryTrie),
        writes to storage lookup *are* immeditaely applied to a trie, generating
        the appropriate trie nodes and and root hash (via the HexaryTrie). The
        writes are *not* persisted to db, until _storage_lookup is explicitly instructed
        to, via :meth:`StorageLookup.commit_to`

        _storage_cache is a cache tied to the state root of the trie. It
        is important that this cache is checked *after* looking for
        the key in _journal_storage, because the cache is only invalidated
        after a state root change. Otherwise, you will see data since the last
        storage root was calculated.

        _locked_changes is a batch database that includes only those values that are
        un-revertable in the EVM. Currently, that means changes that completed in a
        previous transaction.

        Journaling batches writes at the _journal_storage layer, until persist is
        called. It manages all the checkpointing and rollbacks that happen during
        EVM execution.

        In both _storage_cache and _journal_storage, Keys are set/retrieved as the
        big_endian encoding of the slot integer, and the rlp-encoded value.
        """
        self._address = address
        self._storage_lookup = StorageLookup(db, w3, block_identifier, address)
        self._storage_cache = CacheDB(self._storage_lookup)
        self._locked_changes = JournalDB(self._storage_cache)
        self._journal_storage = JournalDB(self._locked_changes)
        self._accessed_slots: Set[int] = set()

        # Track how many times we have cleared the storage. This is journaled
        # in lockstep with other storage changes. That way, we can detect if a revert
        # causes use to revert past the previous storage deletion. The clear count is
        # used as an index to find the base trie from before the revert.
        self._clear_count = JournalDB(MemoryDB({CLEAR_COUNT_KEY_NAME: to_bytes(0)}))

    def get(self, slot: int, from_journal: bool = True) -> int:
        self._accessed_slots.add(slot)
        key = int_to_big_endian(slot)
        lookup_db = self._journal_storage if from_journal else self._locked_changes

        try:
            v = lookup_db[key]
            if v == b"":
                ret = 0
            else:
                ret = int.from_bytes(v, 'big', signed=False)
        except KeyError:
            ret = 0

        if bytes(self._address).hex().startswith('6a9e4959'):
            print(f'{self._address.hex()} storage get {hex(slot)} = {hex(ret)}')

        return ret

    def set(self, slot: int, value: int) -> None:
        if bytes(self._address).hex().startswith('6a9e4959'):
            print(f'{self._address.hex()} storage set {hex(slot)} to {hex(value)}')
        key = int_to_big_endian(slot)
        if value:
            self._journal_storage[key] = value.to_bytes(32, 'big', signed=False)
        else:
            try:
                current_val = self._journal_storage[key]
            except KeyError:
                # deleting an empty key has no effect
                return
            else:
                if current_val != b"":
                    # only try to delete the value if it's present
                    del self._journal_storage[key]

    def delete(self) -> None:
        if bytes(self._address).hex().startswith('6a9e4959'):
            print(f'{self._address.hex()} storage DELETE')

        self.logger.debug2(
            "Deleting all storage in account 0x%s",
            self._address.hex(),
        )
        self._journal_storage.clear()
        self._storage_cache.reset_cache()

        # Look up the previous count of how many times the account has been deleted.
        # This can happen multiple times in one block, via CREATE2.
        old_clear_count = to_int(self._clear_count[CLEAR_COUNT_KEY_NAME])
        new_clear_count = old_clear_count + 1

        # NOTE: below is now meaningless (since I deleted the old code that gets new_clear_count)
        # # Gut check that we have incremented correctly
        # if new_clear_count != old_clear_count + 1:
        #     raise ValidationError(
        #         f"Must increase clear count by one on each delete. Instead, went from"
        #         f" {old_clear_count} -> {new_clear_count} in account"
        #         f" 0x{self._address.hex()}"
        #     )

        # Save the new count, ie~ the index used for a future revert.
        self._clear_count[CLEAR_COUNT_KEY_NAME] = to_bytes(new_clear_count)

    def record(self, checkpoint: JournalDBCheckpoint) -> None:
        self._journal_storage.record(checkpoint)
        self._clear_count.record(checkpoint)

    def discard(self, checkpoint: JournalDBCheckpoint) -> None:
        self.logger.debug2("discard checkpoint %r", checkpoint)
        latest_clear_count = to_int(self._clear_count[CLEAR_COUNT_KEY_NAME])

        if self._journal_storage.has_checkpoint(checkpoint):
            self._journal_storage.discard(checkpoint)
            self._clear_count.discard(checkpoint)
        else:
            # if the checkpoint comes before this account started tracking,
            #    then simply reset to the beginning
            self._journal_storage.reset()
            self._clear_count.reset()
        self._storage_cache.reset_cache()

        reverted_clear_count = to_int(self._clear_count[CLEAR_COUNT_KEY_NAME])

        if reverted_clear_count == latest_clear_count - 1:
            # This revert rewinds past a trie deletion, so roll back to the trie at
            #   that point. We use the clear count as an index to get back to the
            #   old base trie.
            self._storage_lookup.rollback_trie(reverted_clear_count)
        elif reverted_clear_count == latest_clear_count:
            # No change in the base trie, take no action
            pass
        else:
            # Although CREATE2 permits multiple creates and deletes in a single block,
            #   you can still only revert across a single delete. That's because delete
            #   is only triggered at the end of the transaction.
            raise ValidationError(
                f"This revert has changed the clear count in an invalid way, from"
                f" {latest_clear_count} to {reverted_clear_count}, in"
                f" 0x{self._address.hex()}"
            )

    def commit(self, checkpoint: JournalDBCheckpoint) -> None:
        if self._journal_storage.has_checkpoint(checkpoint):
            self._journal_storage.commit(checkpoint)
            self._clear_count.commit(checkpoint)
        else:
            # if the checkpoint comes before this account started tracking,
            #    then flatten all changes, without persisting
            self._journal_storage.flatten()
            self._clear_count.flatten()

    def lock_changes(self) -> None:
        if self._journal_storage.has_clear():
            self._locked_changes.clear()
        self._journal_storage.persist()

    def make_storage_root(self) -> None:
        self.lock_changes()
        self._locked_changes.persist()

    def _validate_flushed(self) -> None:
        """
        Will raise an exception if there are some changes made since the last persist.
        """
        journal_diff = self._journal_storage.diff()
        if len(journal_diff) > 0:
            raise ValidationError(
                "StorageDB had a dirty journal when it needed to be "
                f"clean: {journal_diff!r}"
            )

    def get_accessed_slots(self) -> FrozenSet[int]:
        return frozenset(self._accessed_slots)

    @property
    def has_changed_root(self) -> bool:
        return self._storage_lookup.has_changed_root

    def get_changed_root(self) -> Hash32:
        return self._storage_lookup.get_changed_root()

    def persist(self, db: DatabaseAPI) -> None:
        self._validate_flushed()
        if self._storage_lookup.has_changed_root:
            self._storage_lookup.commit_to(db)
