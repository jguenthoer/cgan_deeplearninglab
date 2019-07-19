import numpy as np
import torch
from torch import optim
from torch.nn import functional as F
import batchloader
import cv2
import os
import network_dropout as network
from tensorboard_evaluation import Evaluation
from datetime import datetime


#device = torch.device("cpu")
device = torch.device("cuda")

def saveimage(image, path):
    # cast to uint8
    if type(image) == torch.Tensor:
        image = image.detach()
        image = image.cpu()
    image = np.array(image, dtype =np.uint8)
    # if image comes from CNN, correct order of axis
    if image.shape ==(3, 218, 178): image = np.moveaxis(image, 0, 2)

    cv2.imwrite(path, image)

    
def starttrain(gen,dis):
    
    genopt = optim.Adam(gen.parameters(), lr=5e-4)
    disopt = optim.Adam(dis.parameters(), lr=5e-4)
    lossfunc = torch.nn.BCEWithLogitsLoss()
    path = '../dataset/celeba'
    batchsize = 32
    lambdaL1 = 2
    results = './results/%s/' % datetime.now().strftime("%Y%m%d-%H%M%S")
    os.mkdir(results)
    
    gen_loss = []
    dis_loss_real = []
    dis_loss_fake = []
    loader = batchloader.BatchLoader(path)
    fake = torch.ones(batchsize,1).to(device)
    real = torch.zeros(batchsize,1).to(device)
    tensorboard_eval = Evaluation('./tensorboard', 'cGanFaces_L1', ["dis_loss_real", "diss_loss_fake", "gen_loss", "dis_loss_real_val", "diss_loss_fake_val", "gen_loss_val"])
    tot = 0
    dis_loss_real_val = 0
    dis_loss_fake_val = 0
    gen_loss_val = 0
    dis_loss_real = 0
    dis_loss_fake = 0
    gen_loss = 0
    train_gen = True
    for epoch in range(1, 200):
        print(epoch)
        val_loader = loader.getvalidationbatch(batchsize)
        for idx, (images, att, lmks) in enumerate(loader.gettrainingbatch(batchsize)):
            # add random noise to attributes
            images = torch.tensor(images, device=device, dtype= torch.float)
            att = torch.tensor(att, device=device, dtype= torch.float)
            lmks = torch.tensor(lmks, device=device, dtype= torch.float)
            

            # train discriminator on batch of real images
            if not train_gen:

                noise = torch.rand(batchsize,1).to(device)
                noise /= 10 
                disopt.zero_grad()
                pred_dis = dis((images/127.5)-1)
                loss = lossfunc(pred_dis, real+noise)
                loss.backward()
                disopt.step()
                dis_loss_real = loss.item()
            
            # train discriminator on batch of fake images


                noise = torch.rand(batchsize,1).to(device)
                noise /= 10 
                disopt.zero_grad()
                fake_img = gen(att, lmks) 
                pred_dis = dis(fake_img)
                loss = lossfunc(pred_dis, fake-noise)
                loss.backward()
                disopt.step()
                dis_loss_fake = loss.item()
                
            # train generator
            if train_gen:

                genopt.zero_grad()
                fake_img = gen(att, lmks)
                pred_dis = dis(fake_img)
                loss = lossfunc(pred_dis, real) + lambdaL1*F.l1_loss((fake_img+1)*127.5, images)
                loss.backward()
                #print(gen.linear1.weight.grad.mean())
                genopt.step()
                gen_loss = loss.item()
                

                                
            #validate    
            if idx%100 == 0:
                dis.eval()
                gen.eval()
                images, att, lmks = next(val_loader, (None,None))
                if type(images) != np.ndarray:
                    val_loader = loader.getvalidationbatch(batchsize)
                    images, att, lmks = next(val_loader, (None,None))
                    
                images = torch.tensor(images, device=device, dtype= torch.float)
                att = torch.tensor(att, device=device, dtype= torch.float)
                lmks = torch.tensor(lmks, device=device, dtype= torch.float)
                
                
                pred_dis = dis((images/127.5)-1)
                loss = lossfunc(pred_dis, real)
                dis_loss_real_val = loss.item()
                
                fake_img = gen(att, lmks) 
                pred_dis = dis(fake_img)
                loss = lossfunc(pred_dis, fake)
                dis_loss_fake_val = loss.item()
                
                loss =lossfunc(pred_dis, real) + lambdaL1*F.l1_loss((fake_img+1)*127.5, images)
                gen_loss_val = loss.item()
                if idx%2000 == 0:
                    image = images[0]
                    saveimage(image, results+str(tot)+'orginal.png')
                    fakeimage = gen(att[0].view(1,40), lmks[0].view(1,10))
                    fakeimage = fakeimage.squeeze()
                    saveimage((fakeimage+1)*127.5, results+str(tot)+'fake.png')
                    
                dis.train()
                gen.train()
            if idx%25 == 0:
                train_gen = not train_gen

                
            #write batch data
            epdata = {"dis_loss_real": dis_loss_real, "diss_loss_fake":dis_loss_fake, "gen_loss":gen_loss,
                      "dis_loss_real_val": dis_loss_real_val, "diss_loss_fake_val":dis_loss_fake_val, "gen_loss_val":gen_loss_val}
            tensorboard_eval.write_episode_data(tot, epdata)
            tot +=1                

        
        torch.save(gen.state_dict(), results+'gen'+str(tot)+'.pt')
        torch.save(dis.state_dict(), results+'dis'+str(tot)+'.pt')
        
def main():
    gen = network.Generator(droprate = 0.2).to(device)
    dis = network.Discriminator().to(device)
    #gen.load_state_dict(torch.load('gen_neu.pt'))
    #dis.load_state_dict(torch.load('dis_neu.pt'))
    

    starttrain(gen,dis)

if __name__ == '__main__':
	main()           
