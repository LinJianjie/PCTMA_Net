import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import torch.nn.parallel
import torch.optim as optim
from pointComNet.models.latent_GAN import *


class LatentGANTrain:
    def __int__(self, LatentGAN, AutoEncoder, lr, epoch):
        self.optimizer_G = optim.Adam(LatentGAN.geneartor.parameters(), lr=lr, betas=(0.9, 0.999))
        self.optimizer_D = optim.Adam(LatentGAN.discriminator.parameters(), lr=lr, betas=(0.9, 0.999))
        self.scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_G, epoch=epoch, eta_min=1e-3)
        self.scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_D, epoch=epoch, eta_min=1e-3)
        self.LatentGAN = LatentGAN
        self.autoEncoder = AutoEncoder

    def train_single_epoch(self, train_loader, device, n_geneartor_update, n_dim, mu, siga):
        for i, real_data in enumerate(train_loader):
            real_data = real_data.to(device)
            ## Training discriminator
            self.optimizer_D.zero_grad()
            self.LatentGAN.discriminator.train()
            d_real_prob, d_real_logit = self.LatentGAN.discriminator(real_data)
            z = generator_noise_distribution(real_data.shape[0], n_dim, mu, siga)
            with torch.no_grad():
                self.LatentGAN.generator.eval()
                g_logit = self.LatentGAN.generator(z)
            d_synthetic_prob, d_synthetic_logit = self.LatentGAN.discriminator(g_logit)
            loss_d = self.LatentGAN.loss_d(d_synthetic_logit, d_real_logit)
            loss_d.backward()
            self.optimizer_D.step()
            self.scheduler_D.step()

            # training Geneator
            if i % n_geneartor_update == 0:
                z = generator_noise_distribution(real_data.shape[0], n_dim, mu, siga)
                self.LatentGAN.generator.train()
                g_logit = self.LatentGAN.generator(z)
                with torch.no_grad():
                    self.LatentGAN.discriminator.eval()
                    d_synthetic_prob_2, d_synthetic_logit_2 = self.LatentGAN.discriminator(g_logit)
                loss_g = self.LatentGAN.loss_g(d_synthetic_logit_2)
                loss_g.backward()
                self.optimizer_G.step()
                self.scheduler_G.step()

    def eval_epoch(self, test_loader, device, n_geneartor_update, n_dim, mu, siga):
        self.LatentGAN.generator.eval()
        self.LatentGAN.discriminator.eval()
        for i, real_data in enumerate(test_loader):
            real_data = real_data.to(device)
            z = generator_noise_distribution(real_data.shape[0], n_dim, mu, siga)
            with torch.no_grad():
                g_logit = self.LatentGAN.generator(z)
                d_synthetic_prob, d_synthetic_logit = self.LatentGAN.discriminator(g_logit)
                d_real_prob, d_real_logit = self.LatentGAN.discriminator(real_data)
                loss_d = self.LatentGAN.loss_d(d_synthetic_logit, d_real_logit)
                loss_g = self.LatentGAN.loss_g(d_synthetic_logit)
                generated_data = self.autoEncoder.decoder(g_logit)
