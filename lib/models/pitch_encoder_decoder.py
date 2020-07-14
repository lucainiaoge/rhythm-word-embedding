#author: Lu Tongyu

from data_loader.Nottingham_database_preprocessor import *
from data_loader.Nottingham_database_preprocessor_util import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_pitchs_octaves_chromas():
    pitches = [i for i in range(-3,128)]
    octaves = [pitch2octave(p) for p in pitches]
    chromas = [pitch2pitchclass(p) for p in pitches]

    print(pitches)
    print(octaves)
    print(chromas)

    pitches_one_hot = [[] for i in range(-3,128)]
    octaves_one_hot = [[] for i in range(-3,128)]
    chromas_one_hot = [[] for i in range(-3,128)]

    for i,_ in enumerate(pitches):
        pitches_one_hot[i] = pitch2onehot(pitches[i])
        octaves_one_hot[i] = octave2onehot(octaves[i])
        chromas_one_hot[i] = pitchclass2onehot(chromas[i])
    return pitches,octaves,chromas,pitches_one_hot,octaves_one_hot,chromas_one_hot

class PitchDataset(Dataset):
    def __init__(self,pitches_one_hot,octaves_one_hot,chromas_one_hot):
        self.pitches_one_hot = torch.tensor(pitches_one_hot)
        self.octaves_one_hot = torch.tensor(octaves_one_hot)
        self.chromas_one_hot = torch.tensor(chromas_one_hot)
    def __len__(self):
        return len(self.pitches_one_hot)
    def __getitem__(self, index):
        this_pitch = self.pitches_one_hot[index]
        this_octave = self.octaves_one_hot[index]
        this_chroma = self.chromas_one_hot[index]
        return this_pitch,this_octave,this_chroma

class PitchEncoder(nn.Module):
    def __init__(self, one_hot_size=130, h_dim=40, z_dim=20):
        super(PitchEncoder, self).__init__()
        self.fc1 = nn.Linear(one_hot_size, h_dim)
        self.fc_encoder_mu = nn.Linear(h_dim, z_dim)
        self.fc_encoder_logvar = nn.Linear(h_dim, z_dim)
        
    def encode(self, x):
        h = F.relu(self.fc1(x))
        mu = self.fc_encoder_mu(h)
        logvar = self.fc_encoder_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var


class PitchDecoder(nn.Module):
    def __init__(self, one_hot_size=130, h_dim=40, z_dim=20):
        super(PitchDecoder, self).__init__()
        self.fc_decoder = nn.Linear(z_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, one_hot_size)
        self.softmax = nn.Softmax()
    
    def forward(self, z):
        h = F.relu(self.fc_decoder(z))
        x_reconst = self.softmax(self.fc2(h))
        return x_reconst


class OctaveClassifier(nn.Module):
    def __init__(self, input_dim=20, out_dim=12):
        super(OctaveClassifier, self).__init__()

        self.fc1 = nn.Linear(input_dim, out_dim)
        self.softmax = nn.Softmax()
    
    def forward(self, z):
        out = self.softmax(self.fc1(z))
        return out

class ChromaClassifier(nn.Module):
    def __init__(self, input_dim=20, out_dim=14):
        super(ChromaClassifier, self).__init__()

        self.fc1 = nn.Linear(input_dim, out_dim)
        self.softmax = nn.Softmax()
    
    def forward(self, z):
        out = self.softmax(self.fc1(z))
        return out


def dataset_sampler(epoch,max_size):
    return int(min(max(epoch,30)/3,max_size))
def VAE_sampler(epoch,max_epoch):
    s = 1-epoch/max_epoch*0.5
    if epoch>max_epoch-100:
        s=1
    return s


def build_pitch_encoder_decoder_model(z_dim=10,hid_dim=80,one_hot_size=130,octave_out_dim=12,chroma_out_dim=14, lr=1e-3):
    global device
    
    model_PitchEncoder = PitchEncoder(one_hot_size, hid_dim, z_dim).to(device)
    model_PitchDecoder = PitchDecoder(one_hot_size, hid_dim, z_dim).to(device)
    model_OctaveClassifier = OctaveClassifier(z_dim, octave_out_dim).to(device)
    model_ChromaClassifier = ChromaClassifier(z_dim, chroma_out_dim).to(device)
    
    optimizer_PitchEncoder = torch.optim.Adam(model_PitchEncoder.parameters(), lr)
    optimizer_PitchDecoder = torch.optim.Adam(model_PitchDecoder.parameters(), lr)
    optimizer_OctaveClassifier = torch.optim.Adam(model_OctaveClassifier.parameters(), lr)
    optimizer_ChromaClassifier = torch.optim.Adam(model_ChromaClassifier.parameters(), lr)

    pitch_VAE_package = {'model_PitchEncoder':model_PitchEncoder,\
                         'model_PitchDecoder':model_PitchDecoder,\
                         'model_OctaveClassifier':model_OctaveClassifier,\
                         'model_ChromaClassifier':model_ChromaClassifier,\
                         'optimizer_PitchEncoder':optimizer_PitchEncoder,\
                         'optimizer_PitchDecoder':optimizer_PitchDecoder,\
                         'optimizer_OctaveClassifier':optimizer_OctaveClassifier,\
                         'optimizer_ChromaClassifier':optimizer_ChromaClassifier}

    return pitch_VAE_package
    # return model_PitchEncoder,model_PitchDecoder,model_OctaveClassifier,model_ChromaClassifier, \
    #     optimizer_PitchEncoder,optimizer_PitchDecoder,optimizer_OctaveClassifier,optimizer_ChromaClassifier



def train_pitch_encoder_decoder(pitch_VAE_package,pitches_one_hot,octaves_one_hot,chromas_one_hot,n_epoch=500,use_VAE=True):
    criterion = nn.BCELoss(reduction='sum')

    model_PitchEncoder = pitch_VAE_package['model_PitchEncoder']
    model_PitchDecoder = pitch_VAE_package['model_PitchDecoder']
    model_OctaveClassifier = pitch_VAE_package['model_OctaveClassifier']
    model_ChromaClassifier = pitch_VAE_package['model_ChromaClassifier']
    optimizer_PitchEncoder = pitch_VAE_package['optimizer_PitchEncoder']
    optimizer_PitchDecoder = pitch_VAE_package['optimizer_PitchDecoder']
    optimizer_OctaveClassifier = pitch_VAE_package['optimizer_OctaveClassifier']
    optimizer_ChromaClassifier = pitch_VAE_package['optimizer_ChromaClassifier']

    model_PitchEncoder.train()
    model_PitchDecoder.train()
    model_OctaveClassifier.train()
    model_ChromaClassifier.train()

    max_size = len(pitches_one_hot)

    pitch_dataset = PitchDataset(pitches_one_hot[0:10],octaves_one_hot[0:10],chromas_one_hot[0:10])
    pitch_dataloader = DataLoader(pitch_dataset, batch_size=1, shuffle=True)

    for epoch in range(n_epoch):
        if random.random()>0.6:
            data_range = dataset_sampler(epoch,max_size)
            pitch_dataset = PitchDataset(pitches_one_hot[0:data_range],octaves_one_hot[0:data_range],chromas_one_hot[0:data_range])
            pitch_dataloader = DataLoader(pitch_dataset, batch_size=1, shuffle=True)
            print('Epoch number %d, change dataset loader range to %d'%(epoch,data_range))

        for i, (this_pitch,this_octave,this_chroma) in enumerate(pitch_dataloader):
            this_pitch = this_pitch.float().to(device)
            this_octave = this_octave.float().to(device)
            this_chroma = this_chroma.float().to(device)
            # train VAE
            z, mu, log_var = model_PitchEncoder(this_pitch)
            x_reconst = model_PitchDecoder(mu)

            reconst_loss_VAE = criterion(x_reconst, this_pitch)
            if random.random()>VAE_sampler(epoch,n_epoch) and use_VAE:
                kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                loss_VAE = reconst_loss_VAE + kl_div
                if (i+1) % 10 == 0:
                    print('kl_div: {:.4f};'.format(kl_div.item()))
            else:
                loss_VAE = reconst_loss_VAE

            optimizer_PitchEncoder.zero_grad()
            optimizer_PitchDecoder.zero_grad()
            loss_VAE.backward()
            optimizer_PitchEncoder.step()
            optimizer_PitchDecoder.step()

            # train Encoder and OctaveClassifier
            if random.random()>0.4:
                z, mu, log_var = model_PitchEncoder(this_pitch)
                octave_pred = model_OctaveClassifier(z)

                loss_octave = criterion(octave_pred, this_octave)
                optimizer_PitchEncoder.zero_grad()
                optimizer_OctaveClassifier.zero_grad()
                loss_octave.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model_OctaveClassifier.parameters(), 1)
                optimizer_PitchEncoder.step()
                optimizer_OctaveClassifier.step()
                if (i+1) % 10 == 0:
                    print('loss_octave: {:.4f};'.format(loss_octave.item()))

            # train Encoder and ChromaClassifier
            if random.random()>0.4:
                z, mu, log_var = model_PitchEncoder(this_pitch)
                chroma_pred = model_ChromaClassifier(z)

                loss_chroma = criterion(chroma_pred, this_chroma)
                optimizer_PitchEncoder.zero_grad()
                optimizer_ChromaClassifier.zero_grad()
                loss_chroma.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model_ChromaClassifier.parameters(), 1)
                optimizer_PitchEncoder.step()
                optimizer_ChromaClassifier.step()
                if (i+1) % 10 == 0:
                    print('loss_chroma: {:.4f};'.format(loss_chroma.item()))

            # stack the loss
            if (i+1) % 10 == 0:
                print ("Epoch[{}/{}], Step [{}/{}], Reconst Loss VAE: {:.4f}" 
                    .format(epoch+1, n_epoch, i+1, len(pitch_dataloader), reconst_loss_VAE.item()))
        
    pitch_VAE_package = {'model_PitchEncoder':model_PitchEncoder,\
                         'model_PitchDecoder':model_PitchDecoder,\
                         'model_OctaveClassifier':model_OctaveClassifier,\
                         'model_ChromaClassifier':model_ChromaClassifier,\
                         'optimizer_PitchEncoder':optimizer_PitchEncoder,\
                         'optimizer_PitchDecoder':optimizer_PitchDecoder,\
                         'optimizer_OctaveClassifier':optimizer_OctaveClassifier,\
                         'optimizer_ChromaClassifier':optimizer_ChromaClassifier}
    return pitch_VAE_package


def load_model_ckpt(model, load_model_path):
    print(f'Load model from {load_model_path}')
    model.load_state_dict(torch.load(load_model_path))
    return model

def load_pitch_VAE_models(pitch_VAE_model_package,load_model_path,n_epoch=500):
    pitch_VAE_model_package['model_PitchEncoder'] = \
    load_model_ckpt(pitch_VAE_model_package['model_PitchEncoder'], f'{load_model_path}/model_PitchEncoder_{n_epoch}.ckpt')
    pitch_VAE_model_package['model_PitchDecoder'] = \
    load_model_ckpt(pitch_VAE_model_package['model_PitchDecoder'], f'{load_model_path}/model_PitchDecoder_{n_epoch}.ckpt')
    pitch_VAE_model_package['model_OctaveClassifier'] = \
    load_model_ckpt(pitch_VAE_model_package['model_OctaveClassifier'], f'{load_model_path}/model_OctaveClassifier_{n_epoch}.ckpt')
    pitch_VAE_model_package['model_ChromaClassifier'] = \
    load_model_ckpt(pitch_VAE_model_package['model_ChromaClassifier'], f'{load_model_path}/model_ChromaClassifier_{n_epoch}.ckpt')
    return pitch_VAE_model_package

def save_models(pitch_VAE_model_package,store_model_path,n_epoch=500):
    model_PitchEncoder = pitch_VAE_model_package['model_PitchEncoder']
    model_PitchDecoder = pitch_VAE_model_package['model_PitchDecoder']
    model_OctaveClassifier = pitch_VAE_model_package['model_OctaveClassifier']
    model_ChromaClassifier = pitch_VAE_model_package['model_ChromaClassifier']
    torch.save(model_PitchEncoder.state_dict(), f'{store_model_path}/model_PitchEncoder_{n_epoch}.ckpt')
    torch.save(model_PitchDecoder.state_dict(), f'{store_model_path}/model_PitchDecoder_{n_epoch}.ckpt')
    torch.save(model_OctaveClassifier.state_dict(), f'{store_model_path}/model_OctaveClassifier_{n_epoch}.ckpt')
    torch.save(model_ChromaClassifier.state_dict(), f'{store_model_path}/model_ChromaClassifier_{n_epoch}.ckpt')


#pitch2zp
#return tensor
def encode_pitch(pitch_1_hot,encoder):
    encoder.eval()
    if type(pitch_1_hot)!=torch.Tensor:
        global device
        this_pitch = torch.LongTensor(pitch_1_hot).float().to(device)
    else:
    	this_pitch = pitch_1_hot
    z, mu, log_var = encoder(this_pitch)
    return z, mu, log_var

#zp2pitch or zp2octave or zp2chroma
#return tensor
def translate_z(z_tensor,decoder):
    decoder.eval()
    if len(z_tensor.shape)==1:
        dim_to_argmax=0
    else:
        dim_to_argmax=1
    pred = decoder(z_tensor)
    num = torch.argmax(pred,dim=dim_to_argmax)
    return pred,num

def pitch_sampler(pitch_z,model_PitchEncoder,model_PitchDecoder):
    decoded_pitch_vec,decoded_pitch = translate_z(pitch_z,model_PitchDecoder)
    pitch_num = decoded_pitch.tolist()
    if type(pitch_num)==list:
        pitch_1_hot=[pitch2onehot(pitch_num[i]) for i in len(pitch_num)]
    else:
        pitch_1_hot = [pitch2onehot(pitch_num)]
    z_resample,_,_ = encode_pitch(pitch_1_hot,model_PitchEncoder)
    print(pitch_1_hot)
    print(z_resample)
    return z_resample,pitch_num,pitch_1_hot
