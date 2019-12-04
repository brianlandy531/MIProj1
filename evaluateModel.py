from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import datetime
import os.path
 
#globals

#some defaults
globalLeaky =False
globalRelu=.2

def randLabel(val, largeSoft, softLabels):
    #print(val)
    

    
    
    if(softLabels):
        
        
        
        if(largeSoft):       
            if(val==0):
                return random.randint(0,30)/100
            else:
                return random.randint(70,120)/100
        else:
        
            if(val==0):
                return random.randint(0,10)/100
            else:
                return random.randint(90,100)/100
    else:
        return val

def adjustLR(epoch):

    if(epoch<50):
        return 5*lr
    elif(epoch<100):
        return 2*lr
    elif(epoch<200):
        return lr
    

#if __name__ == '__main__':
#    torch.multiprocessing.freeze_support()
if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    #torch.multiprocessing.set_start_method('spawn')
    print('loop')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='cifar10 | lsun | mnist |imagenet | folder | lfw | fake')
    parser.add_argument('--dataroot', required=True, help='path to dataset')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--niter', type=int, default=800, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--classes', default='church_outdoor', help='comma separated list of classes for the lsun data set')

    #custom options
    parser.add_argument('--realgets0', action='store_true', help='You want real to be labeled 0?')
    parser.add_argument('--largeSoft', action='store_true', help='big soft labels on both sides, as specified in goodfellow paper')
    parser.add_argument('--allLeaky', action='store_true', help='only use leaky relus')
    parser.add_argument('--softLabels', action='store_true', help='use soft labels on?')
    parser.add_argument('--lrD', type=float, default=0.00005, help='learning rate, default=0.0002')
    parser.add_argument('--historical', action='store_true', help='turn on historical averaging')

    parser.add_argument('--relu_slope', type=float, default=0.2, help='Relu activation leaky Slope')


    parser.add_argument('--long_gen', action='store_true', help='Longer set of filters for generator')


    parser.add_argument('--weightDecay', action='store_true', help='Adam Optimizer regularizer arg')

    parser.add_argument('--DL2Reg', type=float, default=1e-5, help='D regularizer')
    parser.add_argument('--GL2Reg', type=float, default=1e-6, help='G regularizer')

   
   
    opt = parser.parse_args()
    globalLeaky = opt.allLeaky
    globalRelu = opt.relu_slope
   
   
    print(opt)


    #This is to avoid wasting time with combinations that weren't meant to be run together (/yet)
    if(opt.softLabels == False and opt.largeSoft == True):
        print("you can't have large soft labels turned on without turning on the softlabels option first")
        exit()
   
    if(opt.long_gen == True and opt.allLeaky == True):
        print("Long generator networek with leaky relus not set up yet")
        exit()
      
    try:
        os.makedirs(opt.outf)
    except OSError:
        pass
        
        

      
    run_stats = "\\run_stats.txt"
        
    res_file_loc = opt.outf+run_stats

    print(res_file_loc)
   
    res_file = open(res_file_loc, "w")
    res_file.close()
    
    
    
    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    cudnn.benchmark = True

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    if opt.dataset in ['imagenet', 'folder', 'lfw']:
        # folder dataset
        dataset = dset.ImageFolder(root=opt.dataroot,
                                   transform=transforms.Compose([
                                       transforms.Resize(opt.imageSize),
                                       transforms.CenterCrop(opt.imageSize),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))
        nc=3
    elif opt.dataset == 'lsun':
        classes = [ c + '_train' for c in opt.classes.split(',')]
        dataset = dset.LSUN(root=opt.dataroot, classes=classes,
                            transform=transforms.Compose([
                                transforms.Resize(opt.imageSize),
                                transforms.CenterCrop(opt.imageSize),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
        nc=3


    assert dataset
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                             shuffle=True, num_workers=0) #int(opt.workers)) # =0)#

    device = torch.device("cuda:0" if opt.cuda else "cpu")
    ngpu = int(opt.ngpu)
    nz = int(opt.nz)
    ngf = int(opt.ngf)
    ndf = int(opt.ndf)


    # custom weights initialization called on netG and netD
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)


    class Generator(nn.Module):
        def __init__(self, ngpu):
            super(Generator, self).__init__()
            self.ngpu = ngpu
            if(opt.long_gen):
                self.main = nn.Sequential(
                    # input is Z, going into a convolution
                    nn.ConvTranspose2d(     nz,  ngf * 8, 4, 1, 0, bias=False),
                    nn.BatchNorm2d(ngf * 8),
                    nn.ReLU(True),
                    # state size. (ngf*8) x 4 x 4
                    nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(ngf * 4),
                    nn.ReLU(True),
                    # state size. (ngf*4) x 8 x 8
                    nn.ConvTranspose2d(ngf * 4, ngf * 3, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(ngf * 3),
                    nn.ReLU(True),
                    
                    # state size. (ngf*3) x 16 x 16
                    nn.ConvTranspose2d(ngf * 3, ngf * 2, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(ngf * 2),
                    nn.ReLU(True),
                    
                    # state size. (ngf*2) x 24 x 24
                    nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(ngf),
                    nn.ReLU(True),
                    
                    # # state size. (ngf) x 32 x 32
                    # nn.ConvTranspose2d(ngf,     ngf, 4, 1, 1, bias=False),
                    # nn.BatchNorm2d(ngf),
                    # nn.ReLU(True),
                    
                    # state size. (ngf) x 48 x 48
                    nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
                    nn.Tanh()
                    # state size. (nc) x 64 x 64
                )
            
            else:
            
                if(globalLeaky):
                    self.main = nn.Sequential(
                        # input is Z, going into a convolution
                        nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
                        nn.BatchNorm2d(ngf * 8),
                        nn.LeakyReLU(globalRelu),
                        # state size. (ngf*8) x 4 x 4
                        nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                        nn.BatchNorm2d(ngf * 4),
                        nn.LeakyReLU(globalRelu),
                        # state size. (ngf*4) x 8 x 8
                        nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                        nn.BatchNorm2d(ngf * 2),
                        nn.LeakyReLU(globalRelu),
                        # state size. (ngf*2) x 16 x 16
                        nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
                        nn.BatchNorm2d(ngf),
                        nn.LeakyReLU(globalRelu),
                        # state size. (ngf) x 32 x 32
                        nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
                        nn.Tanh()
                        # state size. (nc) x 64 x 64
                    )
                else:
                    self.main = nn.Sequential(
                        # input is Z, going into a convolution
                        nn.ConvTranspose2d(     nz,  ngf * 8, 4, 1, 0, bias=False),
                        nn.BatchNorm2d(ngf * 8),
                        nn.ReLU(True),
                        # state size. (ngf*8) x 4 x 4
                        nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                        nn.BatchNorm2d(ngf * 4),
                        nn.ReLU(True),
                        # state size. (ngf*4) x 8 x 8
                        nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                        nn.BatchNorm2d(ngf * 2),
                        nn.ReLU(True),
                        # state size. (ngf*2) x 16 x 16
                        nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
                        nn.BatchNorm2d(ngf),
                        nn.ReLU(True),
                        # state size. (ngf) x 32 x 32
                        nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
                        nn.Tanh()
                        # state size. (nc) x 64 x 64
                )

        def forward(self, input):
            if input.is_cuda and self.ngpu > 1:
                output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
            else:
                output = self.main(input)
            return output


    netG = Generator(ngpu).to(device)
    netG.apply(weights_init)
    


    class Discriminator(nn.Module):
        def __init__(self, ngpu):
            super(Discriminator, self).__init__()
            self.ngpu = ngpu
            self.main = nn.Sequential(
                # input is (nc) x 64 x 64
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(globalRelu, inplace=True),
                # state size. (ndf) x 32 x 32
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(globalRelu, inplace=True),
                # state size. (ndf*2) x 16 x 16
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(globalRelu, inplace=True),
                # state size. (ndf*4) x 8 x 8
                nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(globalRelu, inplace=True),
                # state size. (ndf*8) x 4 x 4
                nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )

        def forward(self, input):
            if input.is_cuda and self.ngpu > 1:
                output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
            else:
                output = self.main(input)

            return output.view(-1, 1).squeeze(1)


    
    d_loss = []
    d_lossf = []
    d_lossr = []

    g_loss = []
    
    D_G_z1_Arr  = []
    
    D_G_z2_Arr = []
    
    D_x_Arr = []

    netD = Discriminator(ngpu).to(device)
    netD.apply(weights_init)

 
    criterion = nn.BCELoss()

    fixed_noise = torch.randn(opt.batchSize, nz, 1, 1, device=device)
    
    if(opt.realgets0):
        real_label = 0 #1
        fake_label = 1 #0
    else:
        real_label = 1 #1
        fake_label = 0 #0

    # setup optimizer
    
    errDlossSum = 0
    lossAvg = -1.0;
    
    
    if(opt.weightDecay):
    
    
        optimizerD = optim.Adam(netD.parameters(), lr=opt.lrD, betas=(opt.beta1, 0.999), weight_decay=opt.DL2Reg)
        optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.GL2Reg)

    else:
    
        optimizerD = optim.Adam(netD.parameters(), lr=opt.lrD, betas=(opt.beta1, 0.999))
        optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))


    second_cur = 0.0
    sec_last = 0.0
    sec_elap = -1.0

    
    
    modelSavePt = 50
    plotSavePt = 50
    #updateGeneratorEvery = 4

    for epoch in range(opt.niter):
        
        lossAvg = 0
        errDlossSum = 0
        
        #open file
        
        #Load models
        
        modelD = '%s/netD_epoch_%d.pth' % (opt.outf, epoch)
        modelG = '%s/netG_epoch_%d.pth' % (opt.outf, epoch)
        
        
        
        if os.path.isfile(modelD):
            netD.load_state_dict(torch.load(modelD))
            print("D loaded")
            dload=True
        else:
            dload=False
            
        
        if os.path.isfile(modelG):
            netG.load_state_dict(torch.load(modelG))
            print("G loaded")
            gload=True
        else:
            gload=False
            
         # eval model
        
        
        if(dload==False or gload==False):
            #write zeros
            print("Load error writing 0s")
            d_loss.append(0)
            g_loss.append(0)
            d_lossf.append(0)
            d_lossr.append(0)      

            #Store D predictions at end of every epoch
            D_x_Arr.append(0)
            D_G_z2_Arr.append(0)
            D_G_z1_Arr.append(0)     
            
        else:
            
            #run on timte
            for i, data in enumerate(dataloader, 0):
        
        
                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                # train with real
                netD.zero_grad()
                real_cpu = data[0].to(device)
                batch_size = real_cpu.size(0)
                label = torch.full((batch_size,), randLabel(real_label, opt.largeSoft, opt.softLabels), device=device)

                output = netD(real_cpu)
                errD_real = criterion(output, label)
                
                #if(i%updateGeneratorEvery!=0):
                    #if even iteration update D
                errD_real.backward()
                
                
                
                D_x = output.mean().item()

                # train with fake
                noise = torch.randn(batch_size, nz, 1, 1, device=device)
                fake = netG(noise)
                label.fill_(randLabel(fake_label, opt.largeSoft, opt.softLabels))
                output = netD(fake.detach())
                errD_fake = criterion(output, label)
                
            #    if(i%updateGeneratorEvery!=0):
                    #if even iteration update D
                errD_fake.backward()
                
                
                D_G_z1 = output.mean().item()
                errD = errD_real + errD_fake
                
                errDlossSum = errDlossSum + errD_fake
                
                
                
                #if(i%updateGeneratorEvery!=0):
                    #if even iteration update D
                optimizerD.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                netG.zero_grad()
                label.fill_(randLabel(real_label,opt.largeSoft, opt.softLabels))  # fake labels are real for generator cost
                output = netD(fake)
                errG = criterion(output, label)
                errG.backward()
                D_G_z2 = output.mean().item()
                optimizerG.step()
                break
            
            d_loss.append(errD.item())
            g_loss.append(errG.item())
            d_lossf.append(errD_fake.item())
            d_lossr.append(errD_real.item())      

            #Store D predictions at end of every epoch
            D_x_Arr.append(D_x)
            D_G_z2_Arr.append(D_G_z2)
            D_G_z1_Arr.append(D_G_z1)        

        
        res_file = open(res_file_loc, "a")    
        res_file.write('%d %.4f %.4f %.4f %.4f %.4f %.4f %.4f \n' % (epoch, d_loss[-1], d_lossr[-1], d_lossf[-1], g_loss[-1], D_G_z1_Arr[-1], D_G_z2_Arr[-1], D_x_Arr[-1]))            
        res_file.close()
                
            
        