import random, os, glob, shutil

# Config variables
train_test_ratio = 0.9
train_val_ratio = 0.2
base_directory = './images/raw/DR'

# Make train, test directories
train_directory = base_directory + '/train'
val_directory = base_directory + '/val'
test_directory = base_directory + '/test'
if not os.path.exists(train_directory):
	os.makedirs(train_directory)
if not os.path.exists(val_directory):
	os.makedirs(val_directory)
if not os.path.exists(test_directory):
	os.makedirs(test_directory)

files = glob.glob(base_directory + '/*.mat')
random.shuffle(files)

train_size = int(train_test_ratio * len(files))

train_files = files[:train_size]
test_files = files[train_size:]

val_size = int(train_val_ratio * len(train_files))
val_files = train_files[:val_size]
train_files = train_files[val_size:]

for f in train_files:
	shutil.move(f, train_directory)
for f in val_files:
	shutil.move(f, val_directory)
for f in test_files:
	shutil.move(f, test_directory)