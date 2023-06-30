import re

f = open('MinLossWeights.txt', 'r')

txt = f.read()
datas = re.findall('const float (.*?)\[(.*?)\] = {\n(.*?)\n};', txt)

device_declare = ''
host_declare = ''
host_code = 'void SetWeights()\n{\n'
for data in datas:
    name = data[0]
    num = int(data[1])
    value = data[2] + ','
    value = value.replace(' ', '')
    value = value.replace(',', 'f,')
    #value = re.sub('(.*?),(.*?),', lambda matched:'__floats2half2_rn('+matched.group(1)+','+matched.group(2)+'),', value)[:-1]
    #if num % 2 != 0:
    #    value = value + '!'
    #    value = re.sub('(.*?)!', lambda matched:'__float2half2_rn('+matched.group(1)+'),', value)[:-1]
    #num = int((num + 1)/2)
    device_declare += '__constant__ half %s[%d];\n'%(name, num)
    host_declare += 'half %s_[%d] = {\n%s\n};\n'%(name,num,value)
    host_code += "    cudaMemcpyToSymbol(%s, %s_, %d*sizeof(half));\n"%(name, name, num)

host_code += '}'

code = '#pragma once\n#include <cuda_runtime.h>\n#include <cuda_fp16.h>\n\n\n//-----------------------------------------\n\n' + device_declare + '\n//-----------------------------------------\n\n' + host_declare + '\n//-----------------------------------------\n\n' + host_code

f = open('NNWeight.cuh', 'w')
f.write(code)
f.close()









f = open('a.txt', 'r')

txt = f.read()

txt = txt.replace(')', ',)')
txt = txt.replace('(', '(,')

def Div(match):
    try:
        i = int(match.group(1))
        return '%d,'%int((i+1) / 2)
    except:
        return match.group(0)

txt = re.sub('(.*?),', Div, txt)

txt = txt.replace(',)', ')')
txt = txt.replace('(,', '(')

f = open('b.txt', 'w')
f.write(txt)
f.close()
