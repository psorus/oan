from shutil import copyfile as copy

base="C:\\Users\\User\\m\\"

files=[base+"c4\\01\\simple"]
names=["rocbyc"]


for fil,nam in zip(files,names):
  try:
    copy(fil+".png",nam+".png")
  except:pass
  try:
    copy(fil+".pdf",nam+".pdf")
  except:pass
  print("copied", nam)




