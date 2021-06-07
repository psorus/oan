from shutil import copyfile as copy

b1="/work/sk656163/m/c3/od/imgs/"
b1a="/work/sk656163/m/c3/od/imgs2/"
b2="/work/sk656163/m/c3/221/imgs/"
b3="/home/sk656163/m/c1/"


files=[b3+"00/imgs/recqual",b3+"images/reccnew-1",b1a+"emptyscale",b3+"images/reccinv-1",b1a+"trivialptscale",b3+"touse/oneqcd",b3+"touse/onetop",b3+"00/imgs/reloss",b3+"210/encoder","/hpcwork/sk656163/m/c2/229/encoder"]
names=["recqual","lowcompare","emptyscale","recinv","trivialptscale","oneqcd","onetop","reloss","graphencode","denseencode"]

#adding backup
#files+=[b3+"200/imgs/neathistory",b3+"200/imgs/batchhist",b3+"183/imgs/lossbyauc",b3+"29/imgs/lossbyauc",b1+"densescale"]
#names+=["history4","batchhist4","lbalinear","lbaexp","densescale"]



for fil,nam in zip(files,names):
  try:
    copy(fil+".png",nam+".png")
  except:print("failed",fil+".png")
  try:
    copy(fil+".pdf",nam+".pdf")
  except:print("failed",fil+".pdf")
  print("copied", nam)




