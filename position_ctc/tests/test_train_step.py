"""One CTC step on a synthetic crop, so the training shapes stay honest."""

import torch
from torch import nn

from position_ctc.alphabet import TOKEN_TO_ID
from position_ctc.model import MotionCTC


def test_ctc_backward_on_a_short_crop():
    torch.manual_seed(0)
    model = MotionCTC()
    feat = torch.randn(2, 144, 64)
    logits = model(feat)
    assert logits.shape[0] == 2
    assert logits.shape[1] == 16
    targets = torch.tensor(
        [TOKEN_TO_ID["VX+14"], TOKEN_TO_ID["VY+0"], TOKEN_TO_ID["VX+14"], TOKEN_TO_ID["VY+0"]],
        dtype=torch.long,
    )
    loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)
    logp = logits.log_softmax(-1).transpose(0, 1)
    in_len = torch.tensor([16, 16])
    tgt_len = torch.tensor([2, 2])
    loss = loss_fn(logp, targets, in_len, tgt_len)
    loss.backward()
    assert torch.isfinite(loss)
