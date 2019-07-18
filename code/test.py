# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import tkinter as tk
import numpy as np
import batchloader
import cv2
import os
import network
import torch
from tkinter import *
from PIL import Image, ImageTk
import cv2


class Stuff():
    def __init__(self):
        self.lmks = 0
        self.noise = 0
        self.images = 0


class Checkbar(Frame):
   def __init__(self, parent=None, picks=[], side=LEFT, anchor=W):
      Frame.__init__(self, parent)
      self.vars = []
      self.boxes = []
      r = True
      for idx, pick in enumerate(picks):
         var = IntVar()
         chk = Checkbutton(self, text=pick, variable=var)
         chk.grid(row = idx//2, column = idx%2)

         self.vars.append(var)
         self.boxes.append(chk)
   def state(self):
      return map((lambda var: var.get()), self.vars)
  
   def setatt(self, att):
       for idx,var in enumerate(self.vars):
           var.set(att[idx])
           if var.get() == 1:
               self.boxes[idx].select()
           else:
               self.boxes[idx].deselect()
           
if __name__ == '__main__':
   root = Tk()

   device = 'cuda'
   path = '../dataset/celeba'
   loader = batchloader.BatchLoader(path)
   gen = network.Generator().to(device)
   gen.load_state_dict(torch.load('gen35.pt'))
   stuff = Stuff()
   
   scrollbar = Scrollbar(root)
   scrollbar.pack(side=RIGHT, fill=Y)


   attcheck = Checkbar(root, ['5_o_Clock_Shadow',
                             'Arched_Eyebrows',
                             'Attractive',
                             'Bags_Under_Eyes',
                             'Bald',
                             'Bangs',
                             'Big_Lips',
                             'Big_Nose',
                             'Black_Hair',
                             'Blond_Hair',
                             'Blurry',
                             'Brown_Hair',
                             'Bushy_Eyebrows',
                             'Chubby',
                             'Double_Chin',
                             'Eyeglasses',
                             'Goatee',
                             'Gray_Hair',
                             'Heavy_Makeup',
                             'High_Cheekbones',
                             'Male',
                             'Mouth_Slightly_Open',
                             'Mustache',
                             'Narrow_Eyes',
                             'No_Beard',
                             'Oval_Face',
                             'Pale_Skin',
                             'Pointy_Nose',
                             'Receding_Hairline',
                             'Rosy_Cheeks',
                             'Sideburns',
                             'Smiling',
                             'Straight_Hair',
                             'Wavy_Hair',
                             'Wearing_Earrings',
                             'Wearing_Hat',
                             'Wearing_Lipstick',
                             'Wearing_Necklace',
                             'Wearing_Necktie',
                             'Young'])

   attcheck.pack(side=RIGHT,  fill=X)
   #attcheck.config(relief=GROOVE, bd=2)
   
   
   
   test_loader = loader.gettestbatch(1)
   
   images, att, stuff.lmks = next(test_loader, (None,None))
   stuff.noise = torch.rand((1, 78), device = device)

   

   def getnew():
       images, att, stuff.lmks = next(test_loader, (None,None))
       attcheck.setatt(att.squeeze())
       generate()
       
   def getnoise():
       stuff.noise = torch.rand((1, 78), device = device)
     
   def generate():
       
       
       att = list(attcheck.state())
       
       att = torch.tensor(att, device=device, dtype= torch.float).view(1,40)
       lmks = torch.tensor(stuff.lmks, device=device, dtype= torch.float)
       image = gen(att, lmks,stuff.noise)
       print(lmks*255)
       image = image.squeeze()
       
       image = image.detach()
       image = image.cpu()
       image = (image+1)*127.5
       image = np.array(image, dtype =np.uint8)
       image = np.moveaxis(image, 0, 2)
       image = image[...,::-1]
       

       
       
       novi = Toplevel()
       canvas = Canvas(novi, width = 170, height = 200)
       canvas.pack(expand = YES, fill = BOTH)

       image =  ImageTk.PhotoImage(image=Image.fromarray(image))
       canvas.create_image(2,2, image=image,anchor=NW)
       canvas.image = image
       
       
   Button(root, text='New', command=getnew).pack(side=RIGHT)
   Button(root, text='Noise', command=getnoise).pack(side=RIGHT)
   Button(root, text='Generate', command=generate).pack(side=RIGHT)
   root.mainloop()
