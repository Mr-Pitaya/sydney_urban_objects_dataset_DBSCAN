import numpy

def my_dbscan(data, eps, minpts):
    """
    DBSCAN主函数
    data:点云数据集
    eps:阈值距离
    minpts:最小邻域点数量
    return:点云标签
    """    
    labels = [0]*len(data)
    C = 0

    for index in range(0, len(data)):
     
        if not (labels[index] == 0):
            continue
               
        neighborpts = neighbor_find(data, index, eps)
      
        if len(neighborpts) < minpts:
            labels[index] = -1  
        else: 
            C += 1
            grow_cluster(data, labels, index, neighborpts, C, eps, minpts)
   
    return labels


def grow_cluster(data, labels, index, neighborpts, C, eps, minpts):
    """
    发展簇函数
    data:数据集list
    labels:点云标签list
    index:此新簇中核心点的索引
    neighborpts:该核心点的所有临近点
    C:此簇的标签
    eps:阈值距离
    minpts:最小邻域点数量
    """    
    labels[index] = C
   
    i = 0
    while i < len(neighborpts):
        pn = neighborpts[i]

        if labels[pn] == -1:
            labels[pn] = C       
        elif labels[pn] == 0:            
            labels[pn] = C
            pn_neighborpts = neighbor_find(data, pn, eps)

            if len(pn_neighborpts) >= minpts:
                neighborpts = neighborpts + pn_neighborpts
            else:                             
                neighborpts = neighborpts
                
        i += 1 

       
def neighbor_find(data, index, eps):
    """
    搜索领域点函数
    data:数据集list
    index:此新簇中核心点的索引
    eps:阈值距离
    return:该核心点的所有临近点
    """
    neighbors = []
    for pn in range(0, len(data)):
        a = numpy.array(data[index])
        b = numpy.array(data[pn])
       
        if numpy.linalg.norm(a - b) < eps:
            neighbors.append(pn)
           
    return neighbors