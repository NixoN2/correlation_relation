import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t

data = np.array([
    [33.0, 7.5],
    [31.0, 5.7],
    [31.0, 5.4],
    [32.0, 5.8],
    [34.0, 6.8],
    [26.0, 6.2],
    [29.8, 8.0],
    [31.0, 6.1],
    [32.5, 6.8],
    [32.5, 5.6],
    [24.0, 5.0],
    [28.5, 5.0],
    [26.0, 5.4],
    [28.5, 6.7],
    [32.0, 5.3],
    [31.5, 5.5],
    [32.5, 6.4],
    [34.0, 6.3],
    [33.5, 5.5],
    [33.5, 6.0]
])

weights = [x[0] for x in data]
ages = [x[1] for x in data]
weights_set = [x for x in set(weights)]
ages_set = [x for x in set(ages)]
weights_set.sort()
ages_set.sort()

def pcorrelation(x,y):
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    cov = 0
    for i in range(len(x)):
        cov += (x[i]-mean_x)*(y[i]-mean_y)
    std_x = 0
    std_y = 0
    for i in range(len(x)):
        std_x += (x[i]-mean_x) ** 2
    for i in range(len(y)):
        std_y += (y[i]-mean_y) ** 2
    return cov / (std_x ** 0.5 * std_y ** 0.5)

pcorr = pcorrelation(weights,ages)

distribution = [{'weight':x0, 'age':y0,'amount':0}  for x0 in weights_set for y0 in ages_set]

for i in range(len(distribution)):
    for items in data:
        if (distribution[i]['weight'] == items[0] and distribution[i]['age'] == items[1]):
            distribution[i]['amount'] += 1
distribution.sort(key=lambda x: x['weight'])
distribution.sort(key=lambda x: x['age'])

def get_not_null(distribution,value,param):
    return list(filter(lambda x: x['amount'] > 0 and x[param] == value, distribution))

def conditional_mean(distribution,index,ages,weights,param):
    m = 0
    distribution.sort(key=lambda x: x[param])
    value = 0
    k = ''
    if param == 'weight':
        value = weights[index]
        k = 'age'
    elif param == 'age':
        value = ages[index]
        k = 'weight'
    not_null_amount = len(get_not_null(distribution,value,param))
    for i in range(len(distribution)):
        if distribution[i][param] == value:
            m += distribution[i][k] * distribution[i]['amount']
    return m/not_null_amount

#def full_mean(_data):
#    return sum(_data)/len(_data)

def full_mean(distribution,ages,weights,length,param):
    m = 0
    distribution.sort(key=lambda x: x[param])
    if param == 'weight':
        for i in range(len(weights)):
            value = weights[i]
            not_null_amount = len(get_not_null(distribution,value,param))
            m += not_null_amount * value
        return m / length
    elif param == 'age':
        for i in range(len(ages)):
            value = ages[i]
            not_null_amount = len(get_not_null(distribution,value,param))
            m += not_null_amount * value
        return m / length

def conditional_std(distribution,index,ages,weights,param):
    std = 0
    distribution.sort(key=lambda x: x[param])
    k = ''
    if param == 'weight':
        value = weights[index]
        k = 'age'
    elif param == 'age':
        value = ages[index]
        k = 'weight'
    m = conditional_mean(distribution,index,ages,weights,k)
    not_null_amount = len(get_not_null(distribution,value,param))
    for i in range(len(distribution)):
        if distribution[i][param] == value:
            std += distribution[i]['amount'] * (distribution[i][k] - m) ** 2
    return std / not_null_amount

def intergroup_std(distribution, ages,weights,length,param):
    i_std = 0
    distribution.sort(key=lambda x: x[param])
    m = 0
    if param == 'weight':
        m = full_mean(distribution,ages,weights,len(weights),param)
        for i in range(len(weights)):
            value = weights[i]
            c_mean = conditional_mean(distribution,i,ages,weights,'age')
            not_null_amount = len(get_not_null(distribution,value,param))
            i_std += not_null_amount * (c_mean - m) ** 2
        return i_std / length
    elif param == 'age':
        m = full_mean(distribution,ages,weights,len(ages),param)
        for i in range(len(ages)):
            value = weights[i]
            c_mean = conditional_mean(distribution,i,ages,weights,'weight')
            not_null_amount = len(get_not_null(distribution,value,param))
            i_std += not_null_amount * (c_mean - m) ** 2
        return i_std / length

def ingroup(distribution,ages,weights,length,param):
    i_std = 0
    distribution.sort(key=lambda x: x[param])
    if param == 'weight':
        for i in range(len(weights)):
            value=weights[i]
            c_std = conditional_std(distribution,i,ages,weights,param)
            not_null_amount = len(get_not_null(distribution,value,param))
            i_std += not_null_amount * c_std
        return i_std / length
    elif param == 'age':
        for i in range(len(ages)):
            value=ages[i]
            c_std = conditional_std(distribution,i,ages,weights,param)
            not_null_amount = len(get_not_null(distribution,value,param))
            i_std += not_null_amount * c_std
        return i_std / length

inter_std = intergroup_std(distribution,ages_set,weights_set,len(weights_set),'weight')
in_std = ingroup(distribution,ages_set,weights,len(ages),'age')
common_std = inter_std + in_std
correlation_relation = (inter_std / common_std) ** 0.5

t_critical = t.interval(0.95, 18)[1]

print(f'корреляционное отношение: {correlation_relation}')
print(f'коэффициент корреляции Пирсона: {pcorr}')
print(f'Правая критическая точка, уровень значимости 0.05: {t_critical}')

print("Проверка гипотезы:")

t_observed = correlation_relation * (18 / (1 - correlation_relation ** 2)) ** 0.5
if t_observed > t_critical:
    print('Нулевая гипотеза отклонена')
else:
    print('Нулевая гипотеза принята')

print('Оценка расхождения корреляционного отношения и коэффициента корреляции Пирсона:')

y = correlation_relation ** 2 - pcorr ** 2

m = 2 * (y - y ** 2 * (2 - correlation_relation ** 2 - pcorr ** 2)) ** 0.5 / (20) ** 0.5

t_y = y / m

if t_y < t_critical:
    print('Линейная связь')
elif t_y > t_critical:
    print('Нелинейная связь')
