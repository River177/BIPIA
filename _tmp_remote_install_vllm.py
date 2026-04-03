import paramiko, time
host='117.50.185.153'; port=23; user='root'; pw='m94D068i5x3Wbpr2'
ssh=paramiko.SSHClient(); ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy()); ssh.connect(host, port=port, username=user, password=pw, timeout=20)
cmd = "bash -lc 'source /root/venvs/bipia/bin/activate && pip install vllm==0.8.5'"
stdin, stdout, stderr = ssh.exec_command(cmd, timeout=3600)
out = stdout.read().decode('utf-8','replace')
err = stderr.read().decode('utf-8','replace')
print(out)
if err:
    print('STDERR:\n'+err)
ssh.close()
