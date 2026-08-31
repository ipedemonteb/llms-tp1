"""Implementación del Tokenizador Byte-Level BPE (Byte-Pair Encoding).

Este módulo provee la clase `ByteLevelBPETokenizer`, diseñada para entrenar,
guardar, cargar y aplicar tokenización a nivel de subpalabras basada en bytes (Byte-Level BPE)
para arquitecturas Transformer.

Garantiza:
1. Cero tokens desconocidos ([UNK]) gracias al vocabulario base de 256 bytes UTF-8.
2. Formato estándar compatible con PyTorch (input_ids, attention_mask).
3. Configuración personalizable de hiperparámetros (vocab_size, max_length, min_frequency, etc.).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Union

import pandas as pd
import torch
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTokenizer
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer


class ByteLevelBPETokenizer:
    """Tokenizador Byte-Level BPE con soporte completo para PyTorch y HuggingFace Tokenizers."""

    DEFAULT_SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

    def __init__(
        self,
        tokenizer: Optional[Tokenizer] = None,
        max_length: Optional[int] = 128,
        pad_token: str = "[PAD]",
        unk_token: str = "[UNK]",
        cls_token: str = "[CLS]",
        sep_token: str = "[SEP]",
        mask_token: str = "[MASK]",
    ) -> None:
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.cls_token = cls_token
        self.sep_token = sep_token
        self.mask_token = mask_token
        self.max_length = max_length

        if tokenizer is not None:
            self._tokenizer = tokenizer
            self._configure_post_processor()
        else:
            self._tokenizer = Tokenizer(BPE(unk_token=self.unk_token))
            self._tokenizer.pre_tokenizer = ByteLevelPreTokenizer(
                add_prefix_space=False,
                trim_offsets=True
            )
            self._tokenizer.decoder = ByteLevelDecoder()

    @property
    def tokenizer(self) -> Tokenizer:
        """Instancia subyacente de Tokenizer de HuggingFace."""
        return self._tokenizer

    @property
    def vocab_size(self) -> int:
        """Tamaño total del vocabulario."""
        return self._tokenizer.get_vocab_size()

    @property
    def pad_token_id(self) -> int:
        """ID numérico del token [PAD]."""
        return self._tokenizer.token_to_id(self.pad_token)

    @property
    def unk_token_id(self) -> int:
        """ID numérico del token [UNK]."""
        return self._tokenizer.token_to_id(self.unk_token)

    @property
    def cls_token_id(self) -> int:
        """ID numérico del token [CLS] (inicio de secuencia)."""
        return self._tokenizer.token_to_id(self.cls_token)

    @property
    def sep_token_id(self) -> int:
        """ID numérico del token [SEP] (separador / fin de secuencia)."""
        return self._tokenizer.token_to_id(self.sep_token)

    @property
    def mask_token_id(self) -> Optional[int]:
        """ID numérico del token [MASK]."""
        return self._tokenizer.token_to_id(self.mask_token)

    def _configure_post_processor(self) -> None:
        """Configura la inserción automática de tokens especiales [CLS] y [SEP]."""
        cls_id = self.cls_token_id
        sep_id = self.sep_token_id

        if cls_id is not None and sep_id is not None:
            self._tokenizer.post_processor = TemplateProcessing(
                single=f"{self.cls_token} $A {self.sep_token}",
                pair=f"{self.cls_token} $A {self.sep_token} $B {self.sep_token}",
                special_tokens=[
                    (self.cls_token, cls_id),
                    (self.sep_token, sep_id),
                ],
            )

    def train_from_iterator(
        self,
        iterator: Iterable[str],
        vocab_size: int = 2048,
        min_frequency: int = 2,
        special_tokens: Optional[List[str]] = None,
        show_progress: bool = True,
    ) -> ByteLevelBPETokenizer:
        """Entrena el vocabulario BPE a partir de un iterador de strings en memoria."""
        if special_tokens is None:
            special_tokens = self.DEFAULT_SPECIAL_TOKENS

        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=special_tokens,
            initial_alphabet=ByteLevelPreTokenizer.alphabet(),
            show_progress=show_progress,
        )

        self._tokenizer = Tokenizer(BPE(unk_token=self.unk_token))
        self._tokenizer.pre_tokenizer = ByteLevelPreTokenizer(
            add_prefix_space=False,
            trim_offsets=True
        )
        self._tokenizer.decoder = ByteLevelDecoder()

        self._tokenizer.train_from_iterator(iterator, trainer=trainer)
        self._configure_post_processor()
        return self

    def train_from_files(
        self,
        files: Sequence[Union[str, Path]],
        vocab_size: int = 2048,
        min_frequency: int = 2,
        special_tokens: Optional[List[str]] = None,
        show_progress: bool = True,
    ) -> ByteLevelBPETokenizer:
        """Entrena el vocabulario BPE a partir de una lista de archivos de texto."""
        str_files = [str(f) for f in files]
        if special_tokens is None:
            special_tokens = self.DEFAULT_SPECIAL_TOKENS

        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=special_tokens,
            initial_alphabet=ByteLevelPreTokenizer.alphabet(),
            show_progress=show_progress,
        )

        self._tokenizer = Tokenizer(BPE(unk_token=self.unk_token))
        self._tokenizer.pre_tokenizer = ByteLevelPreTokenizer(
            add_prefix_space=False,
            trim_offsets=True
        )
        self._tokenizer.decoder = ByteLevelDecoder()

        self._tokenizer.train(files=str_files, trainer=trainer)
        self._configure_post_processor()
        return self

    def train_from_csv(
        self,
        csv_path: Union[str, Path],
        text_column: str = "text",
        vocab_size: int = 2048,
        min_frequency: int = 2,
        special_tokens: Optional[List[str]] = None,
        show_progress: bool = True,
    ) -> ByteLevelBPETokenizer:
        """Entrena el vocabulario extrayendo la columna de texto de un archivo CSV."""
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"No se encontró el archivo CSV en: {path}")

        df = pd.read_csv(path)
        if text_column not in df.columns:
            raise KeyError(f"Columna '{text_column}' no encontrada en {path}. Columnas disponibles: {list(df.columns)}")

        text_iterator = df[text_column].dropna().astype(str).tolist()
        print(f"🔄 Entrenando Byte-Level BPE sobre {len(text_iterator):,} registros de texto...")
        print(f"   - vocab_size objetivo: {vocab_size}")
        print(f"   - min_frequency: {min_frequency}")

        return self.train_from_iterator(
            iterator=text_iterator,
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=special_tokens,
            show_progress=show_progress,
        )

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
    ) -> Any:
        """Tokeniza un único texto devolviendo el objeto de encoding de HuggingFace."""
        return self._tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def encode_batch(
        self,
        texts: Sequence[str],
        max_length: Optional[int] = None,
        padding: bool = True,
        truncation: bool = True,
        add_special_tokens: bool = True,
        return_tensors: str = "pt",
    ) -> dict[str, Any]:
        """Tokeniza un lote (batch) de textos devolviendo tensores de PyTorch o listas.

        Args:
            texts: Secuencia de strings a tokenizar.
            max_length: Longitud máxima de secuencia. Si es None, usa self.max_length.
            padding: Si True, rellena con [PAD] hasta max_length (o la secuencia más larga del batch).
            truncation: Si True, trunca secuencias que excedan max_length.
            add_special_tokens: Si True, agrega [CLS] al inicio y [SEP] al final.
            return_tensors: Formato de retorno ('pt' para PyTorch Tensors, 'list' para listas Python).

        Returns:
            dict con 'input_ids' y 'attention_mask'.
        """
        eff_max_length = max_length if max_length is not None else self.max_length

        # Configurar truncamiento si está activo
        if truncation and eff_max_length is not None:
            self._tokenizer.enable_truncation(max_length=eff_max_length)
        else:
            self._tokenizer.no_truncation()

        # Configurar padding si está activo
        if padding:
            pad_id = self.pad_token_id if self.pad_token_id is not None else 0
            self._tokenizer.enable_padding(
                direction="right",
                pad_id=pad_id,
                pad_type_id=0,
                pad_token=self.pad_token,
                length=eff_max_length,
            )
        else:
            self._tokenizer.no_padding()

        encodings = self._tokenizer.encode_batch(list(texts), add_special_tokens=add_special_tokens)

        input_ids_list = [enc.ids for enc in encodings]
        attention_mask_list = [enc.attention_mask for enc in encodings]

        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask_list, dtype=torch.long),
            }
        elif return_tensors == "list":
            return {
                "input_ids": input_ids_list,
                "attention_mask": attention_mask_list,
            }
        else:
            raise ValueError(f"return_tensors '{return_tensors}' no soportado. Usa 'pt' o 'list'.")

    def decode(
        self,
        token_ids: Union[List[int], torch.Tensor],
        skip_special_tokens: bool = True,
    ) -> str:
        """Decodifica una secuencia de IDs a su string de texto original."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.detach().cpu().tolist()
        return self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def decode_batch(
        self,
        batch_ids: Union[List[List[int]], torch.Tensor],
        skip_special_tokens: bool = True,
    ) -> List[str]:
        """Decodifica un lote de secuencias de IDs a texto."""
        if isinstance(batch_ids, torch.Tensor):
            batch_ids = batch_ids.detach().cpu().tolist()
        return self._tokenizer.decode_batch(batch_ids, skip_special_tokens=skip_special_tokens)

    def token_to_id(self, token: str) -> Optional[int]:
        """Obtiene el ID numérico correspondiente a un token string."""
        return self._tokenizer.token_to_id(token)

    def id_to_token(self, token_id: int) -> Optional[str]:
        """Obtiene el string de token correspondiente a un ID numérico."""
        return self._tokenizer.id_to_token(token_id)

    def get_vocab(self) -> dict[str, int]:
        """Retorna el diccionario completo de vocabulario {token: id}."""
        return self._tokenizer.get_vocab()

    def save(self, path: Union[str, Path]) -> Path:
        """Guarda la configuración del tokenizador en un archivo JSON."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        self._tokenizer.save(str(save_path))
        print(f"💾 Tokenizador Byte-Level BPE guardado en: {save_path.resolve()}")
        return save_path

    @classmethod
    def from_file(
        cls,
        path: Union[str, Path],
        max_length: Optional[int] = 64,
        pad_token: str = "[PAD]",
        unk_token: str = "[UNK]",
        cls_token: str = "[CLS]",
        sep_token: str = "[SEP]",
        mask_token: str = "[MASK]",
    ) -> ByteLevelBPETokenizer:
        """Carga un tokenizador previamente guardado desde un archivo JSON."""
        load_path = Path(path)
        if not load_path.exists():
            raise FileNotFoundError(f"No se encontró el archivo de tokenizador en: {load_path}")

        tokenizer = Tokenizer.from_file(str(load_path))
        instance = cls(
            tokenizer=tokenizer,
            max_length=max_length,
            pad_token=pad_token,
            unk_token=unk_token,
            cls_token=cls_token,
            sep_token=sep_token,
            mask_token=mask_token,
        )
        return instance


def main():
    """Punto de entrada CLI para entrenar y evaluar el tokenizador BPE."""
    parser = argparse.ArgumentParser(description="Entrena y guarda un Tokenizador Byte-Level BPE para el Transformer.")
    parser.add_argument(
        "--train_file",
        type=str,
        default="resources/datasets/transformer_train.csv",
        help="Ruta al CSV con los datos de entrenamiento (por defecto: resources/datasets/transformer_train.csv).",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default="text",
        help="Nombre de la columna de texto a tokenizar (por defecto: 'text').",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="resources/tokenizer/bpe_tokenizer.json",
        help="Ruta donde se guardará el archivo JSON del tokenizador (por defecto: resources/tokenizer/bpe_tokenizer.json).",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=2048,
        help="Tamaño objetivo del vocabulario (por defecto: 2048).",
    )
    parser.add_argument(
        "--min_frequency",
        type=int,
        default=2,
        help="Frecuencia mínima para fusionar pares de bytes (por defecto: 2).",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=64,
        help="Longitud máxima de secuencia para padding/truncamiento (por defecto: 64).",
    )
    parser.add_argument(
        "--test_sentence",
        type=str,
        default=(
            "Cedar House Steamable Pepperoni Pizza | "
            "Steamable pepperoni pizza in a 10 oz package for online grocery orders. "
            "Listed under frozen and intended for frozen storage. A dependable pick according to reviews. | "
            "Prepared ingredients, Spices, Salt"
        ),
        help="Frase de prueba para validar la tokenización y decodificación.",
    )

    args = parser.parse_args()

    # 1. Instanciar y entrenar
    tokenizer = ByteLevelBPETokenizer(max_length=args.max_length)
    tokenizer.train_from_csv(
        csv_path=args.train_file,
        text_column=args.text_column,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
    )

    # 2. Guardar tokenizador
    tokenizer.save(args.save_path)

    # 3. Validar con frase de prueba
    print("\n" + "=" * 70)
    print("🧪 PRUEBA DE TOKENIZACIÓN Y DECODIFICACIÓN")
    print("=" * 70)
    print(f"Texto original: {args.test_sentence}")

    encoding = tokenizer.encode(args.test_sentence)
    print(f"\nTokens generados ({len(encoding.tokens)} tokens):")
    print(f"  {encoding.tokens}")

    print(f"\nIDs generados ({len(encoding.ids)} IDs):")
    print(f"  {encoding.ids}")

    decoded_text = tokenizer.decode(encoding.ids)
    print(f"\nTexto decodificado (skip_special_tokens=True):")
    print(f"  '{decoded_text}'")

    # 4. Probar encode_batch con tensores de PyTorch
    batch_sample = [
        args.test_sentence,
        (
            "Sunny Basket Ready To Heat Waffles | "
            "Ready to heat waffles in a 6 ct package for online grocery orders. "
            "Listed under frozen and intended for frozen storage. Well liked by regular shoppers. | "
            "Flour, Sugar, Eggs"
        ),
    ]
    batch_output = tokenizer.encode_batch(batch_sample, max_length=args.max_length, return_tensors="pt")

    print("\n" + "=" * 70)
    print("📦 PRUEBA DE BATCH ENCODING (PyTorch Tensors)")
    print("=" * 70)
    print(f"Input IDs Tensor Shape:      {batch_output['input_ids'].shape} (dtype: {batch_output['input_ids'].dtype})")
    print(f"Attention Mask Tensor Shape: {batch_output['attention_mask'].shape} (dtype: {batch_output['attention_mask'].dtype})")
    print(f"Tokens especiales: [PAD]={tokenizer.pad_token_id}, [CLS]={tokenizer.cls_token_id}, [SEP]={tokenizer.sep_token_id}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
