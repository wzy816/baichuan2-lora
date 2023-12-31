import sentencepiece as spm


class BaichuanTokenizer:
    def __init__(
        self,
        vocab_file,
    ):
        self.vocab_file = vocab_file

        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(vocab_file)

        self.add_bos_token = False
        self.add_eos_token = False

        self.unk_token = "<unk>"
        self.unk_token_id = self.sp_model.piece_to_id(self.unk_token)
        self.bos_token = "<s>"
        self.bos_token_id = self.sp_model.piece_to_id(self.bos_token)
        self.eos_token = "</s>"
        self.eos_token_id = self.sp_model.piece_to_id(self.eos_token)
        self.pad_token = "<unk>"
        self.pad_token_id = self.sp_model.piece_to_id(self.pad_token)

    def encode(self, text):
        return self.sp_model.Encode(text)

    def decode(self, tokens):
        return self.sp_model.Decode(tokens)

    def piece_to_id(self, token):
        return self.sp_model.PieceToId(token)

    def id_to_piece(self, index):
        return self.sp_model.IdToPiece(index)

    def vocab_size(self):
        return self.sp_model.get_piece_size()
