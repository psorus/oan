from PIL import Image, ImageFont, ImageDraw, ImageEnhance



def genone(desc,auc,fnam,mul=4):


    source_img = Image.new('RGBA', (300*mul, 100*mul))

    font=ImageFont.truetype("arial.ttf",size=12*mul)
    draw = ImageDraw.Draw(source_img)
    draw.rectangle(((0, 00), (300*mul, 100*mul)), fill="white")
    
    draw.text((20*mul, 10*mul), "Assuming",font=font,fill="black")#, font=ImageFont.truetype("arial.pil"))
    draw.text((50*mul, 28*mul), "The higher "+ str(desc),font=font,fill="darkred")#, font=ImageFont.truetype("arial.pil"))
    draw.text((50*mul, 46*mul), "The more likely this is a top jet.",font=font,fill="black")#, font=ImageFont.truetype("arial.pil"))
    draw.text((20*mul, 64*mul), "You reach an AUC of",font=font,fill="black")#, font=ImageFont.truetype("arial.pil"))
    draw.text((50*mul, 82*mul), str(auc),font=font,fill="darkred")#, font=ImageFont.truetype("arial.pil"))
    
    source_img.save(str(fnam)+".png", "png")
    source_img.save("../imgs/"+str(fnam)+".png", "png")


genone("the loss of the comparison network","0.908","netresa")
genone("the loss of my graph network","0.910","netresb")
genone("R**2","0.915","netresc")
