import tarfile
import numpy as np
import os

# Directory containing the .tar.bz2 files
directory_path = './data'

# List all .tar.bz2 files in the directory
tar_files = [f for f in os.listdir(directory_path) if f.endswith('.tar.bz2')]
# Process each .tar.bz2 file
for tar_filename in tar_files:
    print(f'Reading: {tar_filename}')
    tar_path = os.path.join(directory_path, tar_filename)
    with tarfile.open(tar_path, 'r:bz2') as tar:
        # Process each member in the .tar.bz2 file
        for member in tar.getmembers():
            print(f'Reading: {member.name}')
            # Check if the member is a file and ends with .npy
            if member.isfile() and member.name.endswith('.npy'):
                # Extract the member as a file-like object
                print('Extracting...')
                file = tar.extractfile(member)
                if file:
                    try:
                        # Load the .npy file directly from the tar archive
                        data = np.load(file, allow_pickle=True)
                        # Process the data here
                        for dt in data:
                            print(dt.keys())
                            # If dt is a pandas DataFrame, use dt.head()
                            # Ensure the appropriate method calls based on your data structure
                    except Exception as e:
                        print(f"Error loading {member.name} from {tar_filename}: {e}")
                    finally:
                        file.close()

            if member.isfile() and member.name.endswith('.npz'):
                # Extract the member as a file-like object
                print('Extracting...')
                file = tar.extractfile(member)
                if file:
                    try:
                        # Load the .npy file directly from the tar archive
                        data = np.load(file, allow_pickle=True)
                        # Process the data here
                        print([key for key in list(data.keys())])
                        for key in list(data.keys()):
                            pass
                            # If dt is a pandas DataFrame, use dt.head()
                            # Ensure the appropriate method calls based on your data structure
                    except Exception as e:
                        print(f"Error loading {member.name} from {tar_filename}: {e}")
                    finally:
                        file.close()