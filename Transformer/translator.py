import json
import numpy as np 
import math

def load_embeddings_from_json(filename):

    with open(filename, 'r', encoding='utf-8') as f:
        embeddings_dict = json.load(f)
    return {word: np.array(embedding) for word, embedding in embeddings_dict.items()}

def text_to_embeddings(text, embedding_dict):

    words = text.lower().split()
    
    found_embeddings = []
    
    for word in words:
        if word in embedding_dict:
            found_embeddings.append(embedding_dict[word])

        else: 
            print((f"a palavra '{word}' não está na lista das possíveis palavras"))
            raise Exception(f"Word not in dict.")

    
    return np.array(found_embeddings)

def padding(seq_len, embeddings):

    size = seq_len - embeddings.shape[0]

    if size > 0:
        padding_vectors = np.zeros((size, embeddings.shape[1]))
        embeddings = np.concatenate([embeddings, padding_vectors], axis=0)
    
    return embeddings

def generate_positional_encoding(seq_len, d_model):

    positional_encoding = np.zeros((seq_len, d_model))
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            positional_encoding[pos, i] = np.sin(pos / (100 ** (2 * i / d_model)))
            if i + 1 < d_model:
                positional_encoding[pos, i + 1] = np.cos(pos / (100 ** (2 * i / d_model)))
    return positional_encoding

def generate_attention_weights_npy(d_model, num_heads, head_name):

    d_k = d_model // num_heads  

    
    Wq = np.random.randn(d_model, d_k) / np.sqrt(d_k)
    Wk = np.random.randn(d_model, d_k) / np.sqrt(d_k)
    Wv = np.random.randn(d_model, d_k) / np.sqrt(d_k)
    
    
    np.save(f"{head_name}_Wq.npy", Wq)
    np.save(f"{head_name}_Wk.npy", Wk)
    np.save(f"{head_name}_Wv.npy", Wv)

def generate_weigths_npy(rows,columns,filename):

    W = np.random.rand(rows,columns) 
    np.save(f"{filename}.npy", W)

def softmax(A, inpt_seq):
   
    D = np.full(A.shape, 0, dtype=float)

    B = np.full(A.shape, -np.inf, dtype=float)
    B[:inpt_seq, :inpt_seq] = A[:inpt_seq, :inpt_seq]

    for row in range(B.shape[0]):
        e_x = np.exp(B[row] - np.max(B[row]))
        D[row] = e_x / e_x.sum()
    
    D[inpt_seq:,:] = 0
    return D

def attention_head(Wq,Wk,Wv,A,d_model,num_heads,inpt_seq):
    Q = np.dot(A,Wq)
    K = np.dot(A,Wk)
    V = np.dot(A,Wv)

    Context = np.dot(Q,np.transpose(K))
    Scaled_context = Context/ math.sqrt(d_model/num_heads)

    Attetion_weights = softmax(Scaled_context, inpt_seq) 

    return np.dot(Attetion_weights,V)

def zero_padding(A, inpt_seq):
    seq_len = A.shape[0]

    for i in range(inpt_seq, seq_len):
        for j in range(A.shape[1]):
            A[i, j] = 0  

    return A

def initialize_layernorm_weights(d_model, gamma_file, beta_file):
    
    gamma = np.ones(d_model)
    beta = np.zeros(d_model)

    np.save(f'{gamma_file}.npy',gamma)
    np.save(f'{beta_file}.npy',beta)

def layerNorm(A,gamma,beta):

    for row in A:
        mean = np.mean(row)
        values_mean = row - mean
        s = np.sum(values_mean ** 2)
        v = s / A.shape[1]
        std_deviation = math.sqrt(v)

        with np.errstate(invalid='ignore'):
            new_row = (row - mean)/ std_deviation
    
    for column in range(A.shape[1]):
        A[:, column] *= gamma[column]
        A[:, column] += beta

def ReLU(A):
    for i in range(A.shape[0]):
        for ii in range(A.shape[1]):
            if A[i,ii] < 0:
                A[i,ii] = 0
            
