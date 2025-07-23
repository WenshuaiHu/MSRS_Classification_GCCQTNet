import matplotlib.pyplot as plt

def plot_result(data, data_path, num_result, epoch=0):#https://blog.csdn.net/u010329292/article/details/129597962, https://blog.csdn.net/qq_45100200/article/details/130761143
    fig = plt.figure(figsize = (12,10))
    plt.imshow(data, cmap = 'nipy_spectral')
    plt.colorbar()
    plt.axis('off')
    
    plt.savefig(data_path+ "{}".format(num_result) + '/result_{}.jpg'.format(epoch))
    #plt.show()#
    #spectral.imshow(data.astype(int), figsize(12,10))
    #plt.pause(0)
    plt.close()
    #'''