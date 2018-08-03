import time
import week5_cnn as w

print('\n')
print('\n')
print('\n')

tic = time.time()

w.run()

toc = time.time()

time_elapsed = toc - tic
str_time = "0 ms"

if time_elapsed < 1:
    str_time = str(1000 * time_elapsed) + " ms"
elif time_elapsed < 60:
    str_time = str(time_elapsed) + " sec"
else:
    str_time = str_time(time_elapsed // 60.0) + " mins"

print("\n\nTime taken to complete the operation = ", str_time)
print('\n')
print('\n')
