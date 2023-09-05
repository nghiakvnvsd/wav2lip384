import torch
from torch import nn
from torch.nn import functional as F
import math

from .conv import Conv2dTranspose, Conv2d, nonorm_Conv2d

class Wav2Lip(nn.Module):
    def __init__(self, audio_encoder=None):
        super(Wav2Lip, self).__init__()

        self.face_encoder_blocks = nn.ModuleList([
            
            nn.Sequential(Conv2d(6, 8, kernel_size=7, stride=1, padding=3, act="leaky")), # 384, 384
            
            nn.Sequential(Conv2d(8, 16, kernel_size=3, stride=2, padding=1, act="leaky"), # 192, 192
            Conv2d(16, 16, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),
            Conv2d(16, 16, kernel_size=3, stride=1, padding=1, residual=True, act="leaky")),

            nn.Sequential(Conv2d(16, 32, kernel_size=3, stride=2, padding=1, act="leaky"), # 96, 96
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True, act="leaky")),

            nn.Sequential(Conv2d(32, 64, kernel_size=3, stride=2, padding=1, act="leaky"),    # 48, 48
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True, act="leaky")),

            nn.Sequential(Conv2d(64, 128, kernel_size=3, stride=2, padding=1, act="leaky"),   # 24, 24
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True, act="leaky")),

            nn.Sequential(Conv2d(128, 256, kernel_size=3, stride=2, padding=1, act="leaky"),       # 12, 12
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True, act="leaky")),

            nn.Sequential(Conv2d(256, 512, kernel_size=3, stride=2, padding=1, act="leaky"),     # 6, 6
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),),


            ###################
            # Modified blocks
            ##################
            nn.Sequential(Conv2d(512, 1024, kernel_size=3, stride=2, padding=1, act="leaky"),       # 3, 3
            Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),
            Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, residual=True, act="leaky")),

            nn.Sequential(Conv2d(1024, 1024, kernel_size=3, stride=1, padding=0, act="leaky"),     # 1, 1
            Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0, act="leaky")),])
            ##################


        if audio_encoder is None:
            self.audio_encoder = nn.Sequential(
                Conv2d(1, 32, kernel_size=3, stride=1, padding=1, act="leaky"),
                Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),
                Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),

                Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1, act="leaky"),
                Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),
                Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),

                Conv2d(64, 128, kernel_size=3, stride=3, padding=1, act="leaky"),
                Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),
                Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),

                Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1, act="leaky"),
                Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),
                Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),

                ###################
                # Modified blocks
                ##################
                Conv2d(256, 512, kernel_size=3, stride=1, padding=1, act="leaky"),
                Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),
                Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),

                Conv2d(512, 1024, kernel_size=3, stride=1, padding=0, act="relu"),
                Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0, act="relu"))
                    ##################
        else:
            self.audio_encoder = audio_encoder

        self.audio_refine = nn.Sequential(
            Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0, act="leaky"),
            Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0, act="leaky"))

        self.face_decoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0, act="leaky"),),  # + 1024

            ###################
            # Modified blocks
            ##################
            nn.Sequential(Conv2dTranspose(2048, 1024, kernel_size=3, stride=1, padding=0, act="leaky"), # 3,3
            Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),), # + 1024

            nn.Sequential(Conv2dTranspose(2048, 1024, kernel_size=3, stride=2, padding=1, output_padding=1, act="leaky"),
            Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),
            Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),), # 6, 6  + 512

            nn.Sequential(Conv2dTranspose(1536, 768, kernel_size=3, stride=2, padding=1, output_padding=1, act="leaky"),
            Conv2d(768, 768, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),
            Conv2d(768, 768, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),), # 12, 12  + 256

            nn.Sequential(Conv2dTranspose(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1, act="leaky"),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),), # 24, 24  + 128
            ##################

            nn.Sequential(Conv2dTranspose(640, 256, kernel_size=3, stride=2, padding=1, output_padding=1, act="leaky"),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),), # 48, 48  + 64

            nn.Sequential(Conv2dTranspose(320, 128, kernel_size=3, stride=2, padding=1, output_padding=1, act="leaky"),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),), # 96, 96  + 32

            nn.Sequential(Conv2dTranspose(160, 64, kernel_size=3, stride=2, padding=1, output_padding=1, act="leaky"),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),), # 192, 192  + 16
            
            nn.Sequential(Conv2dTranspose(80, 32, kernel_size=3, stride=2, padding=1, output_padding=1, act="leaky"),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),),]) # 384, 384  + 8

        self.output_block = nn.Sequential(Conv2d(40, 16, kernel_size=3, stride=1, padding=1, act="leaky"),
            nn.Conv2d(16, 3, kernel_size=1, stride=1, padding=0),
            nn.Tanh())

    def freeze_audio_encoder(self):
        for p in self.audio_encoder.parameters():
            p.requires_grad = False

    def new_refine(self):
        self.audio_refine = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 1024),
            nn.Tanh())

    def forward(self, audio_sequences, face_sequences, noise=False):
        if noise:
            return self.forward_with_noise(audio_sequences, face_sequences)

        B = audio_sequences.size(0)

        input_dim_size = len(face_sequences.size())
        if input_dim_size > 4:
            audio_sequences = torch.cat([audio_sequences[:, i] for i in range(audio_sequences.size(1))], dim=0)
            face_sequences = torch.cat([face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0)

        audio_embedding = self.audio_encoder(audio_sequences) # B, 512, 1, 1

        audio_embedding = audio_embedding.detach()
        audio_embedding = self.audio_refine(audio_embedding)

        feats = []
        x = face_sequences
        for f in self.face_encoder_blocks:
            x = f(x)
            feats.append(x)

        x = audio_embedding
        for f in self.face_decoder_blocks:
            x = f(x)
            try:
                x = torch.cat((x, feats[-1]), dim=1)
            except Exception as e:
                print(x.size())
                print(feats[-1].size())
                raise e

            feats.pop()

        x = self.output_block(x)

        if input_dim_size > 4:
            x = torch.split(x, B, dim=0) # [(B, C, H, W)]
            outputs = torch.stack(x, dim=2) # (B, C, T, H, W)

        else:
            outputs = x

        return outputs

    def forward_with_noise(self, noise, face_sequences):

        B = face_sequences.size(0)
        input_dim_size = len(face_sequences.size())
        if input_dim_size > 4:
            face_sequences = torch.cat([face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0)

        feats = []
        x = face_sequences
        for f in self.face_encoder_blocks:
            x = f(x)
            feats.append(x)

        x = noise
        for f in self.face_decoder_blocks:
            x = f(x)
            try:
                x = torch.cat((x, feats[-1]), dim=1)
            except Exception as e:
                print(x.size())
                print(feats[-1].size())
                raise e

            feats.pop()

        x = self.output_block(x)

        if input_dim_size > 4:
            x = torch.split(x, B, dim=0) # [(B, C, H, W)]
            outputs = torch.stack(x, dim=2) # (B, C, T, H, W)

        else:
            outputs = x

        return outputs

    def face_features(self, face_sequences):
        input_dim_size = len(face_sequences.size())
        if input_dim_size > 4:
            face_sequences = torch.cat([face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0)

        feats = []
        x = face_sequences
        for f in self.face_encoder_blocks:
            x = f(x)
            feats.append(x)

        return feats

    def forward_with_feat(self, audio_sequences, feats):
        B = audio_sequences.size(0)

        input_dim_size = len(audio_sequences.size())
        if input_dim_size > 4:
            audio_sequences = torch.cat([audio_sequences[:, i] for i in range(audio_sequences.size(1))], dim=0)

        audio_embedding = self.audio_encoder(audio_sequences) # B, 512, 1, 1

        audio_embedding = audio_embedding
        audio_embedding = self.audio_refine(audio_embedding)

        x = audio_embedding
        for f in self.face_decoder_blocks:
            x = f(x)
            try:
                x = torch.cat((x, feats[-1]), dim=1)
            except Exception as e:
                print(x.size())
                print(feats[-1].size())
                raise e

            feats.pop()

        x = self.output_block(x)

        if input_dim_size > 4:
            x = torch.split(x, B, dim=0) # [(B, C, H, W)]
            outputs = torch.stack(x, dim=2) # (B, C, T, H, W)

        else:
            outputs = x

        return outputs

class Wav2Lip_disc_qual(nn.Module):
    def __init__(self):
        super(Wav2Lip_disc_qual, self).__init__()
        self.face_encoder_blocks = nn.ModuleList([
            nn.Sequential(nonorm_Conv2d(3, 16, kernel_size=7, stride=1, padding=3)), # 192, 384

            nn.Sequential(nonorm_Conv2d(16, 32, kernel_size=5, stride=2, padding=2), # 192, 192
            nonorm_Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            nonorm_Conv2d(32, 32, kernel_size=5, stride=1, padding=2)),

            nn.Sequential(nonorm_Conv2d(32, 64, kernel_size=5, stride=2, padding=2), # 96, 96
            nonorm_Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nonorm_Conv2d(64, 64, kernel_size=5, stride=1, padding=2)),

            nn.Sequential(nonorm_Conv2d(64, 128, kernel_size=5, stride=2, padding=2),    # 48, 48
            nonorm_Conv2d(128, 128, kernel_size=5, stride=1, padding=2),
            nonorm_Conv2d(128, 128, kernel_size=5, stride=1, padding=2)),

            nn.Sequential(nonorm_Conv2d(128, 256, kernel_size=5, stride=2, padding=2),   # 24, 24
            nonorm_Conv2d(256, 256, kernel_size=5, stride=1, padding=2),
            nonorm_Conv2d(256, 256, kernel_size=5, stride=1, padding=2)),

            nn.Sequential(nonorm_Conv2d(256, 512, kernel_size=3, stride=2, padding=1),    # 12, 12
            nonorm_Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nonorm_Conv2d(512, 512, kernel_size=3, stride=1, padding=1)),

            nn.Sequential(nonorm_Conv2d(512, 512, kernel_size=3, stride=2, padding=1),     # 6, 6
            nonorm_Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nonorm_Conv2d(512, 512, kernel_size=3, stride=1, padding=1),),

           # Modified Blocks
            nn.Sequential(nonorm_Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),     # 3, 3
            nonorm_Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),),

            nn.Sequential(nonorm_Conv2d(1024, 1024, kernel_size=3, stride=1, padding=0, norm=False),     # 1, 1
            nonorm_Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0, norm=False)),])

        self.binary_pred = nn.Sequential(nn.Conv2d(1024, 1, kernel_size=1, stride=1, padding=0),
                                         nn.Sigmoid())
        self.label_noise = .0
        self.bce_loss = nn.BCELoss()

    def get_lower_half(self, face_sequences):
        return face_sequences[:, :, face_sequences.size(2)//2:]

    def to_2d(self, face_sequences):
        B = face_sequences.size(0)
        face_sequences = torch.cat([face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0)
        return face_sequences

    def perceptual_forward(self, false_face_sequences):
        false_face_sequences = self.to_2d(false_face_sequences)
        false_face_sequences = self.get_lower_half(false_face_sequences)

        false_feats = false_face_sequences
        for f in self.face_encoder_blocks:
            false_feats = f(false_feats)

        pred = self.binary_pred(false_feats).view(len(false_feats), -1)
        target = torch.ones((len(false_feats), 1)).cuda()
        false_pred_loss = F.binary_cross_entropy(pred, target)

        return false_pred_loss

    def forward(self, face_sequences):
        face_sequences = self.to_2d(face_sequences)

        x = face_sequences
        for f in self.face_encoder_blocks:
            x = f(x)

        return self.binary_pred(x).view(len(x), -1)
