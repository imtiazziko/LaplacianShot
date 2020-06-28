from utils.download import download_file_from_google_drive
import os

if __name__ == '__main__':
    id = '1ejqq3vLYksHpUgwgsS6Q5b27m-AUe2jZ'
    name = "models.zip"
    os.chdir('../')
    if not os.path.isdir('./tmp'):
        os.makedirs('./tmp')
    os.chdir('tmp')
    print('Start Download')
    download_file_from_google_drive(id, name)
    print('Finish Download')
    os.system('unzip models.zip')
    if not os.path.isdir('../results'):
        os.makedirs('../results')
    os.chdir('./models/')
    os.system('mv * ../../results/')