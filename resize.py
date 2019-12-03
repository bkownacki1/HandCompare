from PIL import Image
import os

basewidth = 200


def createFileList(myDir, format='.jpg'):
    fileList = []
    print(myDir)
    for (root, dirs, files) in os.walk(myDir, topdown=True):
        for name in files:
            if name.endswith(format):
                fullName = os.path.join(root,name)
                fileList.append(fullName)
    return fileList

myFileList = createFileList('dataset/Hands')
print(myFileList)


for file in myFileList:
    print(file)
    img = Image.open(file)
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((basewidth, hsize))
    outfile = 'dataset/SmallestHands/%s' % (file[14:])
    img.save(outfile)

    width, height = img.size
    format = img.format
    mode = img.mode


