1.基本的读取配置文件
-read(filename) 直接读取ini文件内容
-sections() 得到所有的section，并以列表的形式返回
-options(section) 得到该section的所有option
-items(section) 得到该section的所有键值对
-get(section,option) 得到section中option的值，返回为string类型
-getint(section,option) 得到section中option的值，返回为int类型，还有相应的getboolean()和getfloat() 函数。
2.基本的写入配置文件
-add_section(section) 添加一个新的section
-set( section, option, value) 对section中的option进行设置，需要调用write将内容写入配置文件。


3.基本例子
test.conf

[sec_a] 
a_key1 = 20 
a_key2 = 10 
   
[sec_b] 
b_key1 = 121 
b_key2 = b_value2 
b_key3 = $r 
b_key4 = 127.0.0.1
parse_test_conf.py

import ConfigParser 
cf = ConfigParser.ConfigParser() 
#read config 
cf.read("test.conf") 
# return all section 
secs = cf.sections() 
print 'sections:', secs 
   
opts = cf.options("sec_a") 
print 'options:', opts 
   
kvs = cf.items("sec_a") 
print 'sec_a:', kvs 
   
#read by type 
str_val = cf.get("sec_a", "a_key1") 
int_val = cf.getint("sec_a", "a_key2") 
   
print "value for sec_a's a_key1:", str_val 
print "value for sec_a's a_key2:", int_val 
   
#write config 
#update value 
cf.set("sec_b", "b_key3", "new-$r") 
#set a new value 
cf.set("sec_b", "b_newkey", "new-value") 
#create a new section 
cf.add_section('a_new_section') 
cf.set('a_new_section', 'new_key', 'new_value') 
   
#write back to configure file 
cf.write(open("test.conf", "w"))
得到终端输出：
sections: ['sec_b', 'sec_a'] 
options: ['a_key1', 'a_key2'] 
sec_a: [('a_key1', "i'm value"), ('a_key2', '22')] 
value for sec_a's a_key1: i'm value 
value for sec_a's a_key2: 22
更新后的test.conf
[sec_b] 
b_newkey = new-value 
b_key4 = 127.0.0.1 
b_key1 = 121 
b_key2 = b_value2 
b_key3 = new-$r 
   
[sec_a] 
a_key1 = i'm value 
a_key2 = 22 
   
[a_new_section] 
new_key = new_value
4.Python的ConfigParser Module中定义了3个类对INI文件进行操作。分别是RawConfigParser、ConfigParser、SafeConfigParser。RawCnfigParser是最基础的INI文件读取类，ConfigParser、SafeConfigParser支持对%(value)s变量的解析。 
设定配置文件test2.conf
[portal] 
url = http://%(host)s:%(port)s/Portal 
host = localhost 
port = 8080
使用RawConfigParser：
import ConfigParser 
  
cf = ConfigParser.RawConfigParser() 
  
print "use RawConfigParser() read" 
cf.read("test2.conf") 
print cf.get("portal", "url") 
  
print "use RawConfigParser() write" 
cf.set("portal", "url2", "%(host)s:%(port)s") 
print cf.get("portal", "url2")

得到终端输出：
use RawConfigParser() read 
http://%(host)s:%(port)s/Portal 
use RawConfigParser() write 
%(host)s:%(port)s

改用ConfigParser：

import ConfigParser 
  
cf = ConfigParser.ConfigParser() 
  
print "use ConfigParser() read" 
cf.read("test2.conf") 
print cf.get("portal", "url") 
  
print "use ConfigParser() write" 
cf.set("portal", "url2", "%(host)s:%(port)s") 
print cf.get("portal", "url2")
得到终端输出：
use ConfigParser() read 
http://localhost:8080/Portal 
use ConfigParser() write 
localhost:8080

改用SafeConfigParser：

import ConfigParser 
  
cf = ConfigParser.SafeConfigParser() 
  
print "use SafeConfigParser() read" 
cf.read("test2.conf") 
print cf.get("portal", "url") 
  
print "use SateConfigParser() write" 
cf.set("portal", "url2", "%(host)s:%(port)s") 
print cf.get("portal", "url2")
得到终端输出(效果同ConfigParser)：
use SafeConfigParser() read 
http://localhost:8080/Portal 
use SateConfigParser() write 
localhost:8080
