from utils.callbacks import LossHistory

loss_history    = LossHistory("logs/")

file = open('logs/loss_2022_01_31_09_56_19/epoch_loss_2022_01_31_09_56_19.txt')
loss = []
while 1:
  line = file.readline()
  if not line:
    break
  loss.append(float(line))
file.close()
print(len(loss))
file = open('logs/loss_2022_01_31_09_56_19/epoch_val_loss_2022_01_31_09_56_19.txt')
loss1 = []
while 1:
  line = file.readline()
  if not line:
    break
  loss1.append(float(line))
print(len(loss1))

file = open('logs/loss_2022_01_31_22_02_37/epoch_loss_2022_01_31_22_02_37.txt')
while 1:
  line = file.readline()
  if not line:
    break
  loss.append(float(line))
file.close()
print(len(loss))
file = open('logs/loss_2022_01_31_22_02_37/epoch_val_loss_2022_01_31_22_02_37.txt')
while 1:
  line = file.readline()
  if not line:
    break
  loss1.append(float(line))
print(len(loss1))

file = open('logs/loss_2022_02_01_09_32_26/epoch_loss_2022_02_01_09_32_26.txt')
while 1:
  line = file.readline()
  if not line:
    break
  loss.append(float(line))
file.close()
print(len(loss))
file = open('logs/loss_2022_02_01_09_32_26/epoch_val_loss_2022_02_01_09_32_26.txt')
while 1:
  line = file.readline()
  if not line:
    break
  loss1.append(float(line))
print(len(loss1))

for i in range(100):
  print(i)
  loss_history.append_loss(loss[i], loss1[i])