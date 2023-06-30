def read_file(filepath):
    with open(filepath) as fp:
        content=fp.read();
    return content

table = read_file('./table.txt')

rpnn = read_file('./Log_RPNN.txt')
rpnn = [i.split('  ') for i in rpnn.split('\n')]
mrpnn = read_file('./Log_MRPNN.txt')
mrpnn = [i.split('  ') for i in mrpnn.split('\n')]

rpnn_zs = read_file('./Log_RPNN_zs.txt')
rpnn_zs = [i.split('  ') for i in rpnn_zs.split('\n')]
mrpnn_zs = read_file('./Log_MRPNN_zs.txt')
mrpnn_zs = [i.split('  ') for i in mrpnn_zs.split('\n')]

for i in rpnn:
    table = table.replace('{RPNN.' + i[0] + '}', i[1][:4])

for i in mrpnn:
    table = table.replace('{MRPNN.' + i[0] + '}', i[1][:4])

for i in rpnn_zs:
    table = table.replace('{RPNN.zs.' + i[0] + '}', i[1][:4])

for i in mrpnn_zs:
    table = table.replace('{MRPNN.zs.' + i[0] + '}', i[1][:4])

with open('./table_out.txt','w') as wf:
    wf.write(table)