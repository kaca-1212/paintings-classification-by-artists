rm -rf lear-gist-python fftw-3.3.8*

echo "fftw3.3.8 requirement"
echo "   Downloading fftw3.3.8 requirement"
wget http://www.fftw.org/fftw-3.3.8.tar.gz
tar zxvf fftw-3.3.8.tar.gz
cd fftw-3.3.8/
echo "   Building and installing fftw3.3.8 requirement"
./configure --enable-single --enable-shared
make
make install
cd ..
echo "fftw3.3.8 installed"

echo "Installing lear-gist-python"
git clone https://github.com/tuttieee/lear-gist-python.git
cd lear-gist-python/
wget http://lear.inrialpes.fr/src/lear_gist-1.2.tgz
tar zxvf lear_gist-1.2.tgz
python3 setup.py build_ext -I /usr/local/include -L /usr/local/lib
python3 setup.py install
echo "Done"