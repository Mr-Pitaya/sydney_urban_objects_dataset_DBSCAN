import csv
import DBSCAN
import pandas as pd

def data_processing(name,*array):
    """
    打开文件，裁剪数据
    name:文件名,文件格式为csv文件
    array:文件中裁剪的范围
    return:一个列表，单个二级列表为一个数据点的xyz浮点值
    """
    with open(name) as scan:        
        
        return list(map(lambda temp: list(map(lambda x: float(x), temp)), list(map(lambda temp: temp[slice(array[0],array[1])], csv.reader(scan)))))


def coordinate_system(data):
    """
    坐标系矫正
    data:点云数据
    return:校正后数据集
    """
    def func(temp):
        temp[0] = temp[0] * -1
        temp[1] = temp[1] * -1
        temp[2] = temp[2] * -1 + 2
        return temp
    
    return list(map(lambda x: func(x), data))


def raster_map(data,**kwargs):
    """
    创建栅格地图及载入数据
    data:源数据
    kwargew:栅格参数
    return:栅格地图列表字典，栅格坐标key为raster_label，数据点列表key为data
    """
    print('检测范围为: x = [{x_min}, {x_max}], y = [{y_min}, {y_max}]'.format(**kwargs))
    print('建立分辨率为{resolution} * {resolution}的矩形栅格'.format(**kwargs))

       
    def raster_list():
        """
        确定栅格列表左下坐标
        return:栅格列表左下坐标
        """
        x = kwargs['x_min']
        y = kwargs['y_min']
        raster_coordinate_list = []
        while True:
            raster_coordinate_list.append((x, y,))
            x += kwargs['resolution']
            if x >= kwargs['x_max']:
                y += kwargs['resolution']
                x = kwargs['x_min']
            if y >= kwargs['y_max']:
                break
        return raster_coordinate_list

     
    def func(temp):
        """
        定位栅格
        temp:数据点坐标
        return:数据点位于栅格索引
        """
        if temp[0] >= kwargs['x_min'] and temp[0] <= kwargs['x_max'] and temp[1] >= kwargs['y_min'] and temp[1] <= kwargs['y_max']:
            temp_x = temp[0] // kwargs['resolution']
            temp_y = temp[1] // kwargs['resolution']
            return int((kwargs['x_max'] - kwargs['x_min']) * (temp_y - kwargs['y_min']) + (temp_x - kwargs['x_min']))

    raster_map_list = list(map(lambda temp: dict(raster_label = temp, data = []), raster_list()))
    for temp in data:
        if type(func(temp)) is int:
            raster_map_list[func(temp)]['data'].append(temp)
    
    print('已生成栅格地图，栅格总数为{num}'.format(num = len(raster_map_list)))
    return raster_map_list


def raster_map_filter(data, **parameter):
    """
    栅格滤波
    data:数据列表字典
    parameter:过滤参数
    return:字典中删除已过滤点
    """
    raster_filter = list(filter(lambda temp: temp['data'] != [], data))

    for temp in raster_filter:
        temp.update(filter_list = list(map(lambda x: x[2], temp['data'])))

    raster_filter = list(filter(lambda temp: (max(temp['filter_list']) - min(temp['filter_list']) >= parameter['delta_h']), raster_filter))

    for temp in raster_filter:
        temp.update(data = list(filter(lambda x: x[2] > parameter['ground_h'], temp['data'])))

    raster_filter = list(filter(lambda temp: temp['data'] != [], raster_filter))

    for temp in raster_filter:
        temp.pop('filter_list')

    return raster_filter


def raster_clustering(data, parameter):
    """
    进行栅格聚类
    data:栅格数据
    parameter:栅格参数
    return:根据聚类结果排序的栅格数据，聚类标签key为label;标签最大值
    """


    def my_clustering(data,parameter):
        """
        聚类主函数
        data:栅格数据
        parameter:栅格参数
        return:标签列表
        """
        labels = [1] * len(data)
        C = 1
        for index in range(len(data)):
            if labels[index] != 1:
                continue

            neighbor = neighbor_find(data, index, parameter['resolution'])

            if len(neighbor) == 1:
                labels[index] = 0
            else:
                C += 1
                grow_cluster(data, labels, index, neighbor, C, parameter['resolution'])
            
        return labels


    def neighbor_find(data, index, resolution):       
        """
        关联性函数
        data:栅格数据
        index:当前处理栅格数据索引
        resolution:栅格分辨率
        return:关联栅格索引
        """
        nei_list = []
        for nei_index in range(len(data)):
            a = data[index]['raster_label']
            b = data[nei_index]['raster_label']
            if (abs(a[0] - b[0]) in (0,resolution) and 
                abs(a[1] - b[1]) in (0,resolution)):
                nei_list.append(nei_index)

        return nei_list


    def grow_cluster(data, labels, index, neighbor, C, resolution):
        """
        更新簇函数
        data:栅格数据
        labels:当前标签列表
        index:当前处理栅格数据索引
        neighbor:neighbor_find函数处理结果
        C:当前labels列表索引
        resolution:栅格分辨率
        """
        labels[index] = C
        i = 0
        while i < len(neighbor):
            nei_index = neighbor[i]

            if labels[nei_index] == 0:
                labels[nei_index] = C
            elif labels[nei_index] == 1:
                labels[nei_index] = C
                new_neighbor = neighbor_find(data, nei_index, resolution)

                if len(new_neighbor) >= 1:
                    neighbor = neighbor + new_neighbor
                else:
                    neighbor = neighbor

            i += 1
    labels = my_clustering(data,parameter)

    for index,label in enumerate(labels):
        data[index].update(label = label)
    
    return sorted(data, key= lambda temp: temp['label']), max(labels)


def local_dbscan(data, label_max, parameter):
    """
    局部聚类
    data:栅格数据
    label_max:栅格分类总数
    parameter:聚类参数
    return:新的栅格列表字典,k:v为：
        raster_label:栅格左下坐标列表
        data:点云数据
        label:栅格标签
        data_labels:聚类后的标签，噪声点为-1
        result:带标签的数据点，结构为x,y,z,label
        sign_list:栅格内对象的可视化处理
    """
    local_dbscan_list = list(map(lambda x: dict(raster_label = [], data = []), list(range(label_max + 1))))
    for temp in data:
        local_dbscan_list[temp['label']]['raster_label'].append(temp['raster_label'])
        local_dbscan_list[temp['label']]['data'].extend(temp['data'])
        local_dbscan_list[temp['label']].update(label = temp['label'])

    for index, temp in enumerate(local_dbscan_list):

        if temp['data'] != []:
            print('栅格{index}中数据点数量为{num}'.format(index = index, num = len(temp['data'])))
            temp.update(data_labels = DBSCAN.my_dbscan(temp['data'], parameter['eps'], parameter['minpts']), result = [])

            for i,x in enumerate(temp['data']):
                temp['result'].append(x + [temp['data_labels'][i]])
                if max(temp['data_labels']) > 0:
                    temp.update(sign_list = sign(temp['data'],temp['data_labels']))
                else:
                    temp.update(sign_list = [])

            pd.DataFrame(data = temp['result']).to_csv('local_dbscan_{index}.csv'.format(index = index), index= None, header= None, encoding= 'gbk')
            print('已生成局部聚类文件local_dbscan_{index}.csv'.format(index = index))
            if  temp['sign_list'] == []:
                print('该栅格没有有效数据')
            else:
                pd.DataFrame(data = temp['sign_list']).to_csv('visualization_{index}.csv'.format(index = index), index= None, header= None, encoding= 'gbk')
                print('已生成局部聚类可视化结果visualization_{index}.csv'.format(index = index))

    return local_dbscan_list            

def sign(data,index):
    """
    聚类效果可视化
    data:单个栅格中的数据点列表
    index:数据点聚类后的标签列表
    return:可视化点列表
    """
    count = 1
    result_list = []

    while True:
        temp_list = []
        color = 100

        for i ,temp in enumerate(index):
            
            if temp == count:
                temp_list.append(data[i])

        x_min = min(temp_list, key= lambda x: x[0])[0]
        x_max = max(temp_list, key= lambda x: x[0])[0]
        y_min = min(temp_list, key= lambda y: y[1])[1]
        y_max = max(temp_list, key= lambda y: y[1])[1]
        z_min = min(temp_list, key= lambda z: z[2])[2]
        z_max = max(temp_list, key= lambda z: z[2])[2]

        x = x_min 
        while True:
            result_list.append([x, y_min, z_min, color])
            result_list.append([x, y_min, z_max, color])
            result_list.append([x, y_max, z_min, color])
            result_list.append([x, y_max, z_max, color])

            if x >= x_max:
                break

            x += 0.01

        y = y_min 
        while True:
            result_list.append([x_min, y, z_min, color])
            result_list.append([x_min, y, z_max, color])
            result_list.append([x_max, y, z_min, color])
            result_list.append([x_max, y, z_max, color])

            if y >= y_max:
                break

            y += 0.01

        z = z_min 
        while True:
            result_list.append([x_min, y_min, z, color])
            result_list.append([x_min, y_max, z, color])
            result_list.append([x_max, y_min, z, color])
            result_list.append([x_max, y_max, z, color])

            if z >= z_max:
                break

            z += 0.01

        if count >= max(index):
            break
        count +=1

    return result_list