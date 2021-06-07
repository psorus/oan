from shutil import copyfile as copy

b1="C:\\Users\\User\\m\\"
b2="C:\\Users\\User\\Desktop\\"
b3="C:\\Users\\User\\m\\q\\02\\imgs\\"
b4="C:\\Users\\User\\m\\c4\\01\\"


files=[b1+"c4\\01\\simple",b1+"mmt\\q\\10\\imgs\\compare",b2+"Untitled Diagram (3)",b2+"Untitled Diagram (7)",b2+"Untitled Diagram (8)",b3+"02",b3+"04",b3+"06",b3+"add1",b4+"sout\\dist1",b4+"sout\\dist2",b4+"sout\\add3",b4+"sout\\add2",b4+"sout\\add4",b4+"sout\\abc",b1+"rroc\\ittot"]
names=["rocbyc","compare","dia3","dia7","dia8","ani2","ani4","ani6","aniadd","dist1","dist2","add3","add2","add4","abc","ttot"]


for fil,nam in zip(files,names):
  try:
    copy(fil+".png",nam+".png")
  except:print("failed",fil)
  try:
    copy(fil+".pdf",nam+".pdf")
  except:print("failed",fil)
  print("copied", nam)



