import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from onmt.modules import Embeddings
from onmt.encoders import RNNEncoder, TransformerEncoder
from onmt.decoders.decoder import StdRNNDecoder
from onmt.decoders.transformer import TransformerDecoder

from onmt.translate import GNMTGlobalScorer
from onmt.translate.beam import Beam

from dataset import PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX


'''
Model definition
'''
class Net(nn.Module):
    def __init__(self, model_type, encoder, decoder0, decoder1):
        super(Net, self).__init__()
        
        self.type = model_type
        self.encoder = encoder
        self.decoder0 = decoder0
        self.decoder1 = decoder1
        self.generator = None
        
        self.choose_decoder = lambda dec_idx: decoder0 if dec_idx == 0 else decoder1
        
    def forward(self, src, lengths, dec_idx, only_enc=False):
        enc_final, memory_bank = self.encoder(src[1:], lengths-1)
        if only_enc == True:
            if self.type == 'GRU':
                if self.encoder.rnn.bidirectional:
                    enc_final = torch.cat([enc_final[0:enc_final.size(0):2], enc_final[1:enc_final.size(0):2]], 2)
                return enc_final.squeeze(dim=0)
            elif self.type == 'TRANS':
                return enc_final.mean(dim=0)
        
        enc_state = self.choose_decoder(dec_idx).init_decoder_state(src[1:], memory_bank, enc_final)
        decoder_outputs, dec_state, attns = self.choose_decoder(dec_idx)(src, memory_bank, enc_state)
        
        decoded = self.generator(decoder_outputs)
        
        return decoded #decoder_outputs, attns, dec_state
    
    
    def generate(self, src, lengths, dec_idx, max_length=20, beam_size=5, n_best=1):
        assert dec_idx == 0 or dec_idx == 1
        batch_size = src.size(1)
        
        def var(a):
            return torch.tensor(a, requires_grad=False)
        
        def rvar(a):
            return var(a.repeat(1, beam_size, 1))
        
        def bottle(m):
            return m.view(batch_size * beam_size, -1)

        def unbottle(m):
            return m.view(beam_size, batch_size, -1)
        
        def from_beam(beam):
            ret = {"predictions": [],
                   "scores": [],
                   "attention": []}
            for b in beam:
                scores, ks = b.sort_finished(minimum=n_best)
                hyps, attn = [], []
                for i, (times, k) in enumerate(ks[:n_best]):
                    hyp, att = b.get_hyp(times, k)
                    hyps.append(hyp)
                    attn.append(att)
                ret["predictions"].append(hyps)
                ret["scores"].append(scores)
                ret["attention"].append(attn)
            return ret
        
        
        scorer = GNMTGlobalScorer(0, 0, "none", "none")
        
        beam = [Beam(beam_size, n_best=n_best,
                     cuda=self.cuda(),
                     global_scorer=scorer,
                     pad=PAD_IDX,
                     eos=SOS_IDX,
                     bos=EOS_IDX,
                     min_length=0,
                     stepwise_penalty=False,
                     block_ngram_repeat=0)
                for __ in range(batch_size)]
        
        enc_final, memory_bank = self.encoder(src, lengths)
        
        token = torch.full((1, batch_size, 1), SOS_IDX, dtype=torch.long, device=next(self.parameters()).device)
        dec_state = enc_final
        dec_state = self.choose_decoder(dec_idx).init_decoder_state(src, memory_bank, dec_state)
               
        memory_bank = rvar(memory_bank.data)
        memory_lengths = lengths.repeat(beam_size)
        dec_state.repeat_beam_size_times(beam_size)
        
        # unroll
        all_indices = []
        for i in range(max_length):
            if all((b.done() for b in beam)):
                break
                
            inp = var(torch.stack([b.get_current_state() for b in beam]).t().contiguous().view(1, -1))
            inp = inp.unsqueeze(2)
                
            decoder_output, dec_state, attn = self.choose_decoder(dec_idx)(inp, memory_bank, dec_state, memory_lengths=memory_lengths, step=i)
            
            decoder_output = decoder_output.squeeze(0)
            
            out = self.generator(decoder_output).data
            out = unbottle(out)
            
            # beam x tgt_vocab
            beam_attn = unbottle(attn["std"])
            
            for j, b in enumerate(beam):
                b.advance(out[:, j], beam_attn.data[:, j, :memory_lengths[j]])
                dec_state.beam_update(j, b.get_current_origin(), beam_size)
    
        ret = from_beam(beam)
#        ret["src"] = src.transpose(1, 0)

        return ret
    
class Discriminator(nn.Module):
    def __init__(self, ninput, noutput, layers, activation=nn.ReLU(), device=torch.device("cpu")):
        super(Discriminator, self).__init__()
        self.ninput = ninput
        self.noutput = noutput

        layer_sizes = [ninput] + [int(x) for x in layers.split('-')]
        self.layers = []

        for i in range(len(layer_sizes)-1):
            layer = nn.Linear(layer_sizes[i], layer_sizes[i+1]).to(device)
            self.layers.append(layer)
            self.add_module("layer"+str(i+1), layer)

            # No batch normalization in first layer
            if i != 0:
                bn = nn.BatchNorm1d(layer_sizes[i+1]).to(device)
                self.layers.append(bn)
                self.add_module("bn"+str(i+1), bn)

            self.layers.append(activation)
            self.add_module("activation"+str(i+1), activation)

        layer = nn.Linear(layer_sizes[-1], noutput).to(device)
        self.layers.append(layer)
        self.add_module("layer"+str(len(self.layers)), layer)

        self.init_weights()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        x = torch.sigmoid(x)
        return x

    def init_weights(self):
        init_std = 0.02
        for layer in self.layers:
            try:
                layer.weight.data.normal_(0, init_std)
                layer.bias.data.fill_(0)
            except:
                pass
            
            
'''
Model Builder composed of RNN or Transformer

    opt : if GRU,   (vocab_size, embed_dim, rnn_size, num_layers, dec_dropout, bidirectional, disc_layer)
          if TRANS, (vocab_size, embed_dim, trans_size, num_heads, ff_size, enc_dropout, dec_dropout, disc_layer)
'''
def build_model(opt, model_type='GRU', device=torch.device("cpu")):
    assert model_type == 'GRU' or model_type == 'TRANS'
    
    vocab_size = opt[0]
    model_size = opt[2]
    disc_layer = opt[-1]
    bi_gru = False
    if model_type == 'GRU':
        bi_gru = opt[5]
    
    print('Build model...')
    if model_type == 'GRU':
        encoder, decoder0, decoder1 = build_rnn_model(*opt[:-1])
    elif model_type == 'TRANS':
        encoder, decoder0, decoder1 = build_trans_model(*opt[:-1])
    
    # Build Net(= encoder + decoder0 + decoder1).
    model = Net(model_type, encoder, decoder0, decoder1)
        
    generator = nn.Sequential(
        nn.Linear(model_size, vocab_size),
        nn.LogSoftmax(dim=-1))
    
    if model_type == 'TRANS':
        for p in model.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
        for p in generator.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    '''
    if hasattr(model.encoder, 'embeddings'):
        model.encoder.embeddings.load_pretrained_vectors(
            model_opt.pre_word_vecs_enc, model_opt.fix_word_vecs_enc)
    if hasattr(model.decoder1, 'embeddings'):
        model.decoder1.embeddings.load_pretrained_vectors(
            model_opt.pre_word_vecs_dec, model_opt.fix_word_vecs_dec)
    if hasattr(model.decoder2, 'embeddings'):
        model.decoder2.embeddings.load_pretrained_vectors(
            model_opt.pre_word_vecs_dec, model_opt.fix_word_vecs_dec)
    '''
    
    # Add generator to model (this registers it as parameter of model).
    model.generator = generator
    model.to(device)
    
    print('Build discriminator...')
    # Build discriminator
    disc = Discriminator(ninput=model_size, noutput=1, layers=disc_layer, device=device)
    disc.to(device)

    return model, disc

def build_rnn_model(vocab_size, embed_dim, rnn_size, num_layers, dropout_p, bidirectional):
    # Build encoder
    src_embeddings = Embeddings(
        word_vec_size=embed_dim,
        word_vocab_size=vocab_size,
        word_padding_idx=0
    )
    encoder = RNNEncoder("GRU", bidirectional, num_layers, rnn_size, dropout=dropout_p, embeddings=src_embeddings)
    
    tgt_embeddings0 = Embeddings(
        word_vec_size=embed_dim,
        word_vocab_size=vocab_size,
        word_padding_idx=0
    )
    decoder0 = StdRNNDecoder("GRU", bidirectional, num_layers, rnn_size, dropout=dropout_p, embeddings=tgt_embeddings0)
    tgt_embeddings1 = Embeddings(
        word_vec_size=embed_dim,
        word_vocab_size=vocab_size,
        word_padding_idx=0
    )
    tgt_embeddings1 = Embeddings(
        word_vec_size=embed_dim,
        word_vocab_size=vocab_size,
        word_padding_idx=0
    )
    decoder1 = StdRNNDecoder("GRU", bidirectional, num_layers, rnn_size, dropout=dropout_p, embeddings=tgt_embeddings1)
    
    return encoder, decoder0, decoder1

#def build_rnn_model(vocab_size, embed_dim, rnn_size, num_layers, dropout_p, bidirectional):
#    # Build encoder
#    encoder = EncoderRNN(vocab_size, embed_dim, rnn_size, num_layers, bidirectional=bidirectional)
#    
#    # Build decoders
#    decoder0 = DecoderRNN(rnn_size, embed_dim, vocab_size, num_layers, dropout_p)
#    decoder1 = DecoderRNN(rnn_size, embed_dim, vocab_size, num_layers, dropout_p)
#    
#    return encoder, decoder0, decoder1

def build_trans_model(vocab_size, embed_dim, model_dim, num_layers, num_heads, ff_size, enc_dropout, dec_dropout):
    # Build encoders
    src_embeddings = Embeddings(
        word_vec_size=model_dim,
        word_vocab_size=vocab_size,
        word_padding_idx=0,
        position_encoding=True
    )
    encoder = TransformerEncoder(
        num_layers=num_layers, d_model=model_dim, heads=num_heads,
        d_ff=ff_size, dropout=enc_dropout, embeddings=src_embeddings
    )
    
    # Build decoders
    tgt_embeddings0 = Embeddings(
        word_vec_size=model_dim,
        word_vocab_size=vocab_size,
        word_padding_idx=0,
        position_encoding=True
    )
    decoder0 = TransformerDecoder(
        num_layers=num_layers,
        d_model=model_dim,
        heads=num_heads,
        d_ff=ff_size,
        attn_type=None,
        copy_attn=False,
        self_attn_type="scaled-dot",
        dropout=dec_dropout,
        embeddings=tgt_embeddings0
    )
    
    tgt_embeddings1 = Embeddings(
        word_vec_size=model_dim,
        word_vocab_size=vocab_size,
        word_padding_idx=0,
        position_encoding=True
    )
    decoder1 = TransformerDecoder(
        num_layers=num_layers,
        d_model=model_dim,
        heads=num_heads,
        d_ff=ff_size,
        attn_type=None,
        copy_attn=False,
        self_attn_type="scaled-dot",
        dropout=dec_dropout,
        embeddings=tgt_embeddings1
    )
    
    return encoder, decoder0, decoder1











class EncoderRNN(nn.Module):
    def __init__(self, input_size, emb_size, hidden_size, n_layers, bidirectional):
        super(EncoderRNN, self).__init__()
        
        self.input_size = input_size
        self.emb_size = emb_size
        self.hidden_size = int(hidden_size / (2 if bidirectional else 1))
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        
        self.embedding = nn.Embedding(self.input_size, self.emb_size, padding_idx=0)
        self.rnn = nn.GRU(self.emb_size, self.hidden_size, bidirectional=self.bidirectional, num_layers=self.n_layers)
        
    def forward(self, input_seqs, input_lens):
        embedded = self.embedding(input_seqs)
        packed = pack_padded_sequence(embedded, input_lens)
        outputs, hidden = self.rnn(packed)
        outputs, output_lengths = pad_packed_sequence(outputs)
        
        if self.bidirectional:
            hidden = torch.cat([hidden[0:hidden.size(0):2], hidden[1:hidden.size(0):2]], 2)
            
        return hidden, outputs
    
    
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, emb_size, output_size, n_layers, dropout_p):
        super(DecoderRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.emb_size = emb_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        
        self.embedding = nn.Embedding(self.output_size, self.emb_size, padding_idx=0)
        self.rnn = nn.GRU(self.emb_size, self.hidden_size, num_layers=self.n_layers, dropout=dropout_p)
    
    def forward(self, input_seqs, hidden, enc_output):
        embedded = self.embedding(input_seqs)
        output, hidden = self.rnn(embedded, hidden)
        return hidden, output
