from shutil import copyfile as copy

b1="/work/sk656163/m/c3/od/imgs/"
b1a="/work/sk656163/m/c3/od/imgs2/"
b2="/work/sk656163/m/c3/221/imgs/"
b3="/home/sk656163/m/c1/"


files=[b1+"trivscale",b1+"compscale",b1+"compscale_zoom",b1a+"splitscale",b1+"superscale",b2+"meanangle3",b2+"meanpt1",b2+"meanpt7",b3+"00/imgs/aucmapb",b3+"14/imgs/aucmapb",b3+"images/deltaauc_00_14",b1a+"angularscale",b1+"ptscale"]
names=["trivscale","compscale","compscale_zoom","splitscale","superscale","meanangle3","meanpt1","meanpt7","qcdmap","topmap","deltamap","angularscale","ptscale"]

#adding backup
files+=[b3+"200/imgs/neathistory",b3+"200/imgs/batchhist",b3+"183/imgs/lossbyauc",b3+"29/imgs/lossbyauc",b1+"densescale"]
names+=["history4","batchhist4","lbalinear","lbaexp","densescale"]



for fil,nam in zip(files,names):
  try:
    copy(fil+".png",nam+".png")
  except:print("failed",fil+".png")
  try:
    copy(fil+".pdf",nam+".pdf")
  except:print("failed",fil+".pdf")
  print("copied", nam)




