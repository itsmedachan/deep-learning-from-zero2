class Embedding:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.idx = None

    def forward(self, idx):
        W, = self.params
        self.idx = idx
        out = W[idx]
        return out

    def backward(self, dout):
        dW, = self.grads
        dW[...] = 0

        # for i, word_id in enumerate(self.idx):
        #   dW[word_id] += dout[i]
        # もしくは
        np.add.at(dW, self.idx, dout) # pythonのfor文より、Numpyメソッドのほうが一般的に速い

        return None
