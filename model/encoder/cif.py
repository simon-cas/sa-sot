import torch
import torch.nn as nn


class CIF(nn.Module):
    """
    Continuous Integrate-and-Fire (CIF) with boundary tracking
    """

    def __init__(self, threshold=1.0):
        super().__init__()
        self.threshold = threshold

    def forward(self, h, alpha, target_len=None):
        """
        Args:
            h:          (B, T, D)
            alpha:      (B, T)
            target_len: (B,) or None

        Returns:
            c:           (B, N, D) padded CIF outputs
            token_lens:  (B,)
            boundaries:  List[List[(start, end)]], length B
                         frame indices per CIF token
        """
        B, T, D = h.size()
        device = h.device

        # quantity scaling (training)
        if target_len is not None:
            alpha_sum = alpha.sum(dim=1)
            scale = target_len.float() / (alpha_sum + 1e-6)
            alpha = alpha * scale.unsqueeze(1)

        outputs = []
        token_lens = []
        boundaries = []

        for b in range(B):
            integrate = 0.0
            frame_acc = torch.zeros(D, device=device)

            fired = []
            fired_bounds = []

            token_start = None  # start frame index of current token

            for t in range(T):
                a = alpha[b, t]
                h_t = h[b, t]

                if token_start is None and a > 0:
                    token_start = t

                integrate_new = integrate + a

                if integrate_new < self.threshold:
                    integrate = integrate_new
                    frame_acc = frame_acc + a * h_t
                else:
                    # fire
                    remain = self.threshold - integrate
                    frame_acc = frame_acc + remain * h_t

                    fired.append(frame_acc)
                    fired_bounds.append((token_start, t + 1))  # [start, end)

                    # reset
                    integrate = integrate_new - self.threshold
                    frame_acc = integrate * h_t
                    token_start = t if integrate > 0 else None

            # tail handling (optional, but safer)
            if integrate > 1e-4:
                fired.append(frame_acc)
                fired_bounds.append((token_start, T))

            fired = torch.stack(fired, dim=0)  # (N, D)

            outputs.append(fired)
            token_lens.append(fired.size(0))
            boundaries.append(fired_bounds)

        max_len = max(token_lens)
        c = h.new_zeros(B, max_len, D)

        for b, out in enumerate(outputs):
            c[b, : out.size(0)] = out

        token_lens = torch.tensor(token_lens, device=device)

        return c, token_lens, boundaries

