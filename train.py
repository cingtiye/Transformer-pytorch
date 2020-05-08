# -*- coding: utf-8 -*-
import torch
import torch.optim as optim
import time
import torch.nn as nn

def SequenceMask(X, X_len,value=0):
    maxlen = X.size(1)
    mask = torch.arange(maxlen)[None, :].to(X_len.device) < X_len[:, None]
    X[~mask] = value
    return X

class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    def forward(self, pred, label, valid_length):
        # the sample weights shape should be (batch_size, seq_len)
        weights = torch.ones_like(label)
        weights = SequenceMask(weights, valid_length).float()
        self.reduction='none'
        output=super(MaskedSoftmaxCELoss, self).forward(pred.transpose(1,2), label)
        return (output*weights).mean(dim=1)

class NoamOpt:
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer    # 优化器
        self._step = 0                # 步长
        self.warmup = warmup          # warmup_steps
        self.factor = factor          # 学习率因子(就是学习率前面的系数)
        self.model_size = model_size  # d_model
        self._rate = 0                # 学习率

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))

def grad_clipping(params, theta, device):
    """Clip the gradient."""
    norm = torch.tensor([0], dtype=torch.float32, device=device)
    for param in params:
        norm += (param.grad ** 2).sum()
    norm = norm.sqrt().item()
    if norm > theta:
        for param in params:
            param.grad.data.mul_(theta / norm)

def grad_clipping_nn(model, theta, device):
    """Clip the gradient for a nn model."""
    grad_clipping(model.parameters(), theta, device)

# def get_std_opt(model):
#     return NoamOpt(model.src_embed[0].d_model, 2, 4000,
#                    torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

def train(model, data_iter, lr, factor, warmup, num_epochs, device):
    model.to(device)
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer = NoamOpt(model.enc_net.embedding_size, factor, warmup,
                        torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9))
    loss = MaskedSoftmaxCELoss()
    tic = time.time()
    for epoch in range(1, num_epochs + 1):
        l_sum, num_tokens_sum = 0.0, 0.0
        for batch in data_iter:
            optimizer.optimizer.zero_grad()
            X, X_vlen, Y, Y_vlen = [x.to(device) for x in batch]
            Y_input, Y_label, Y_vlen = Y[:, :-1], Y[:, 1:], Y_vlen - 1

            Y_hat, _ = model(X, Y_input, X_vlen)
            l = loss(Y_hat, Y_label, Y_vlen).sum()
            l.backward()

            with torch.no_grad():
                grad_clipping_nn(model, 5, device)
            num_tokens = Y_vlen.sum().item()
            optimizer.step()
            l_sum += l.sum().item()
            num_tokens_sum += num_tokens
        if epoch % 50 == 0:
            print("epoch {0:4d},loss {1:.3f}, time {2:.1f} sec".format(
                epoch, (l_sum / num_tokens_sum), time.time() - tic))
            tic = time.time()

def translate(model, src_sentence, src_vocab, tgt_vocab, max_len, device):
    """Translate based on an encoder-decoder model with greedy search."""
    src_tokens = src_vocab[src_sentence.lower().split(' ')]
    src_len = len(src_tokens)
    if src_len < max_len:
        src_tokens += [src_vocab.pad] * (max_len - src_len)
    enc_X = torch.tensor(src_tokens, device=device)
    enc_valid_length = torch.tensor([src_len], device=device)
    # use expand_dim to add the batch_size dimension.
    enc_outputs = model.enc_net(enc_X.unsqueeze(dim=0), enc_valid_length)
    dec_state = model.dec_net.init_state(enc_outputs, enc_valid_length)
    dec_X = torch.tensor([tgt_vocab.bos], device=device).unsqueeze(dim=0)
    predict_tokens = []
    for _ in range(max_len):
        Y, dec_state = model.dec_net(dec_X, dec_state)
        # The token with highest score is used as the next time step input.
        dec_X = Y.argmax(dim=2)
        py = dec_X.squeeze(dim=0).int().item()
        if py == tgt_vocab.eos:
            break
        predict_tokens.append(py)
    return ' '.join(tgt_vocab.to_tokens(predict_tokens))