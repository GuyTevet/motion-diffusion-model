import torch
import torch.nn as nn


# adapted from action2motion to take inputs of different lengths
class MotionDiscriminator(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_layer, device, output_size=12, use_noise=None):
        super(MotionDiscriminator, self).__init__()
        self.device = device

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_layer = hidden_layer
        self.use_noise = use_noise

        self.recurrent = nn.GRU(input_size, hidden_size, hidden_layer)
        self.linear1 = nn.Linear(hidden_size, 30)
        self.linear2 = nn.Linear(30, output_size)

    def forward(self, motion_sequence, lengths=None, hidden_unit=None):
        # dim (motion_length, num_samples, hidden_size)
        bs, njoints, nfeats, num_frames = motion_sequence.shape
        motion_sequence = motion_sequence.reshape(bs, njoints*nfeats, num_frames)
        motion_sequence = motion_sequence.permute(2, 0, 1)
        if hidden_unit is None:
            # motion_sequence = motion_sequence.permute(1, 0, 2)
            hidden_unit = self.initHidden(motion_sequence.size(1), self.hidden_layer)
        gru_o, _ = self.recurrent(motion_sequence.float(), hidden_unit)

        # select the last valid, instead of: gru_o[-1, :, :]
        out = gru_o[tuple(torch.stack((lengths-1, torch.arange(bs, device=self.device))))]

        # dim (num_samples, 30)
        lin1 = self.linear1(out)
        lin1 = torch.tanh(lin1)
        # dim (num_samples, output_size)
        lin2 = self.linear2(lin1)
        return lin2

    def initHidden(self, num_samples, layer):
        return torch.randn(layer, num_samples, self.hidden_size, device=self.device, requires_grad=False)


class MotionDiscriminatorForFID(MotionDiscriminator):
    def forward(self, motion_sequence, lengths=None, hidden_unit=None):
        # dim (motion_length, num_samples, hidden_size)
        bs, njoints, nfeats, num_frames = motion_sequence.shape
        motion_sequence = motion_sequence.reshape(bs, njoints*nfeats, num_frames)
        motion_sequence = motion_sequence.permute(2, 0, 1)
        if hidden_unit is None:
            # motion_sequence = motion_sequence.permute(1, 0, 2)
            hidden_unit = self.initHidden(motion_sequence.size(1), self.hidden_layer)
        gru_o, _ = self.recurrent(motion_sequence.float(), hidden_unit)

        # select the last valid, instead of: gru_o[-1, :, :]
        out = gru_o[tuple(torch.stack((lengths-1, torch.arange(bs, device=self.device))))]

        # dim (num_samples, 30)
        lin1 = self.linear1(out)
        lin1 = torch.tanh(lin1)
        return lin1


model_path = "./assets/actionrecognition/humanact12_gru.tar"


def load_classifier(input_size_raw, num_classes, device):
    model = torch.load(model_path, map_location=device)
    classifier = MotionDiscriminator(input_size_raw, 128, 2, device=device, output_size=num_classes).to(device)
    classifier.load_state_dict(model["model"])
    classifier.eval()
    return classifier


def load_classifier_for_fid(input_size_raw, num_classes, device):
    model = torch.load(model_path, map_location=device)
    classifier = MotionDiscriminatorForFID(input_size_raw, 128, 2, device=device, output_size=num_classes).to(device)
    classifier.load_state_dict(model["model"])
    classifier.eval()
    return classifier


def test():
    from src.datasets.ntu13 import NTU13
    import src.utils.fixseed  # noqa

    classifier = load_classifier("ntu13", input_size_raw=54, num_classes=13, device="cuda").eval()
    params = {"pose_rep": "rot6d",
              "translation": True,
              "glob": True,
              "jointstype": "a2m",
              "vertstrans": True,
              "num_frames": 60,
              "sampling": "conseq",
              "sampling_step": 1}
    dataset = NTU13(**params)

    from src.models.rotation2xyz import Rotation2xyz
    rot2xyz = Rotation2xyz(device="cuda")
    confusion_xyz = torch.zeros(13, 13, dtype=torch.long)
    confusion = torch.zeros(13, 13, dtype=torch.long)

    for i in range(1000):
        dataset.pose_rep = "xyz"
        data = dataset[i][0].to("cuda")
        data = data[None]

        dataset.pose_rep = params["pose_rep"]
        x = dataset[i][0].to("cuda")[None]
        mask = torch.ones(1, x.shape[-1], dtype=bool, device="cuda")
        lengths = mask.sum(1)

        xyz_t = rot2xyz(x, mask, **params)

        predicted_cls_xyz = classifier(data, lengths=lengths).argmax().item()
        predicted_cls = classifier(xyz_t, lengths=lengths).argmax().item()

        gt_cls = dataset[i][1]

        confusion_xyz[gt_cls][predicted_cls_xyz] += 1
        confusion[gt_cls][predicted_cls] += 1

    accuracy_xyz = torch.trace(confusion_xyz)/torch.sum(confusion_xyz).item()
    accuracy = torch.trace(confusion)/torch.sum(confusion).item()

    print(f"accuracy: {accuracy:.1%}, accuracy_xyz: {accuracy_xyz:.1%}")


if __name__ == "__main__":
    test()
