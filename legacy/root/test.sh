#!/bin/bash
#PBS -q Eduq
#PBS -l select=1:ncpus=1:mem=8gb
#PBS -l walltime=00:10:00

cd $PBS_O_WORKDIR

echo "--- TMP ACCESS TEST START ---"

# 1. $TMPDIR のパスを確認
echo "Current TMPDIR: $TMPDIR"

# 2. $TMPDIR に書き込みができるかテスト (1GBのファイルを作成)
echo "Creating 1GB dummy file in $TMPDIR..."
dd if=/dev/zero of=$TMPDIR/test_file bs=1M count=1024

# 3. $TMPDIR の使用量と空き容量を確認
echo "Disk space in $TMPDIR:"
df -h $TMPDIR

# 4. 書き込んだファイルを削除
rm $TMPDIR/test_file
echo "--- TEST FINISHED ---"