from typing import (
    Any,
)
from eth_typing import Address

import rlp
from rlp.sedes import (
    big_endian_int,
)
from web3 import Web3
from web3.types import (
    BlockIdentifier,
)

from eth.abc import (
    AccountAPI,
)
from eth.constants import (
    BLANK_ROOT_HASH,
    EMPTY_SHA3,
)
from eth.db.journal import JournalDB

from .sedes import (
    hash32,
    trie_root,
)


class Account(rlp.Serializable, AccountAPI):
    """
    RLP object for accounts.
    """

    fields = [
        ("nonce", big_endian_int),
        ("balance", big_endian_int),
        ("storage_root", trie_root),
        ("code_hash", hash32),
    ]

    def __init__(
        self,
        nonce: int = 0,
        balance: int = 0,
        storage_root: bytes = BLANK_ROOT_HASH,
        code_hash: bytes = EMPTY_SHA3,
        **kwargs: Any,
    ) -> None:
        super().__init__(nonce, balance, storage_root, code_hash, **kwargs)

    def __repr__(self) -> str:
        return (
            f"Account(nonce={self.nonce}, balance={self.balance}, "
            f"storage_root=0x{self.storage_root.hex()}, "
            f"code_hash=0x{self.code_hash.hex()})"
        )

class LazyAccount(AccountAPI):
    _nonce: big_endian_int = None
    _balance: big_endian_int = None
    _storage_root: trie_root = None
    _code_hash: hash32 = None

    _web3: Web3
    _block_identifier: BlockIdentifier
    _address: Address

    def __init__(
            self,
            journaldb: JournalDB,
            web3: Web3,
            block_identifier: BlockIdentifier,
            address: Address,
    ) -> None:
        self._journaldb = journaldb
        self._web3 = web3
        self._block_identifier = block_identifier
        self._address = address

    @property
    def nonce(self) -> big_endian_int:
        if self._nonce is None:
            self._nonce = self._web3.eth.get_transaction_count(
                self._address,
                block_identifier=self._block_identifier,
            )
        return self._nonce

    @nonce.setter
    def nonce(self, value: big_endian_int) -> None:
        self._nonce = value


    @property
    def balance(self) -> big_endian_int:
        if self._balance is None:
            self._balance = self._web3.eth.get_balance(
                self._address,
                block_identifier=self._block_identifier,
            )
        return self._balance

    @balance.setter
    def balance(self, value: big_endian_int) -> None:
        self._balance = value

    @property
    def storage_root(self) -> trie_root:
        raise NotImplementedError("Too lazy to implement")
    
    @property
    def code_hash(self) -> hash32:
        raise NotImplementedError("Too lazy to implement")

    def copy(self, **kwargs) -> AccountAPI:
        ret = LazyAccount(
            web3=self._web3,
            block_identifier=self._block_identifier,
            address=self._address,
        )
        if 'nonce' in kwargs:
            print(f'{self._address.hex()} copying nonce={kwargs["nonce"]}')
        ret._nonce = kwargs.get('nonce', self._nonce)
        ret._balance = kwargs.get('balance', self._balance)

        return ret

    def __str__(self) -> str:
        return f'LazyAccount(address={self._address.hex()}; _nonce={self._nonce}; _balance={self._balance})'
