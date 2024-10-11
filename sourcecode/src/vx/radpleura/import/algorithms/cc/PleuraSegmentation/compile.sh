# sudo apt-get install build-essential cmake
# sudo apt-get install libgtk-3-dev
# sudo apt-get install libboost-all-dev

# install itk
# https://itk.org/download/
#
# manual de instalacion
# https://itk.org/Wiki/ITK_Configuring_and_Building_for_Ubuntu_Linux
# comando c dos veces!!!!!!!!!!!!!!!!!!!!

# install dlib
# https://kumarvinay.com/installing-dlib-library-in-ubuntu/
# or
# pip3 install dlib
# rm -rf build && mkdir build
cmake -S . -B build
cmake --build build
