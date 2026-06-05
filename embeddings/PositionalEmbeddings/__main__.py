from . import PositionalEmbeddings


max_seq_len, emb_size = 10, 3
pe = PositionalEmbeddings(
    max_seq_len, 
    emb_size
)

seq_len, start_pos = 5, 3
print(pe.forward(
    seq_len=seq_len,
    start_pos=start_pos
))