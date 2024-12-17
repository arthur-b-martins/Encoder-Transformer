import numpy as np
import translator as tl


seq_len = 16 
d_model = 16
num_heads = 2


embeddings_dict = tl.load_embeddings_from_json('word_embeddings.json')
list_of_words = embeddings_dict.keys()
print(list_of_words)


inpt = input("Digite sua frase de entrada: ")
inpt_embeddings = tl.text_to_embeddings(inpt, embeddings_dict)
inpt_seq = inpt_embeddings.shape[0]
print(f'a quantidade de palavras na frase é {inpt_seq}')
if inpt_seq > 16:
    raise Exception("Sua frase deve ter até 16 palavras.")

inpt_embeddings_padding = tl.padding(seq_len, inpt_embeddings)


positional_encoding = tl.generate_positional_encoding(seq_len, d_model)
A = inpt_embeddings_padding + positional_encoding


A = tl.zero_padding(A,inpt_seq)
print(f'Dimensões A: {A.shape}')
# ------ Aqui nós entramos no mecanismo de atenção ------------------------------
Wq1 = np.load('attention_head1_Wq.npy')
Wk1 = np.load('attention_head1_Wk.npy')
Wv1 = np.load('attention_head1_Wv.npy')
Wq2 = np.load('attention_head2_Wq.npy')
Wk2 = np.load('attention_head2_Wk.npy')
Wv2 = np.load('attention_head2_Wv.npy')
print(f'Dimensões Wq: {Wq1.shape}')

Z1 = tl.attention_head(Wq1, Wk1, Wv1, A, d_model, num_heads, inpt_seq)
print(f'Dimensões Z1: {Z1.shape}')

Z2 = tl.attention_head(Wq2, Wk2, Wv2, A, d_model, num_heads, inpt_seq)
print(f'Dimensões Z2: {Z2.shape}')

Concat_output = np.concatenate([Z1,Z2],axis=1)
print(f'Dimensões Concat_output: {Concat_output.shape}')
Wo = np.load('output_weight.npy')
print(f'Dimensões Wo: {Wo.shape}')

Attention_output = np.dot(Concat_output, Wo)
print(f'Dimensões Output: {Attention_output.shape}')

# ------- Conexão residual ------------------------------------------------------ 

Z = Attention_output + A 
print(f'Dimensões Skip: {Z.shape}')


# ------- Layer Normalization ---------------------------------------------------
gamma1 = np.load('gamma_E_attention.npy')
beta1 = np.load('beta_E_attention.npy')

tl.layerNorm(Z, gamma1, beta1)


# -------- Feed Forward --------------------------------------------------------- 

Wff_e = np.load('feedforward_E_weight.npy')
Bff_e = np.load('feedforward_E_bias.npy')

print('Verifique as dimensões do peso e viés: ')
print(Wff_e.shape)
print(Bff_e.shape)

Zff_e = np.dot(Z,Wff_e) + Bff_e
print("Dimensões da resultante expandida: ")
print(Zff_e.shape)


tl.ReLU(Zff_e)
print(Zff_e)

Wff_r = np.load('feedforward_E_weight2.npy')
Bff_r = np.load('feedforward_E_bias2.npy')
Zff_r = np.dot(Zff_e, Wff_r) + Bff_r
print(Zff_r.shape)

# ---------- Conexão residual ---------------------------------------------------

Encoder_output = Zff_r + Z
print(f'Dimensões Skip: {Z.shape}')

# ---------- Layer Normalization ------------------------------------------------


gamma2 = np.load('gamma_E_feedforward.npy')
beta2 = np.load('beta_E_feedforward.npy')

tl.layerNorm(Encoder_output, gamma2, beta2)

# --------- FIM DO ENCODER -----------------------------------------------------


