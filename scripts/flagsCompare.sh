gcc -Q -O2 --help=optimizers > O2_flags.txt
gcc -Q -O3 --help=optimizers > O3_flags.txt

diff O2_flags.txt O3_flags.txt | grep enabled