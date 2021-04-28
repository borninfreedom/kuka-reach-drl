# VSCode tricks
## [About python extensions](https://zhuanlan.zhihu.com/p/361654489?utm_source=com.miui.notes&utm_medium=social&utm_oi=903420714332332032)

## Resolve a.py in A folder import b.py in B folder
* Add the codes below at the top of a .py file
```python
import os,inspect
current_dir=os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.chdir(current_dir)
import sys
sys.path.append('../')
```
## Add header template in .py files
* Select FIle -> Preference -> User Snippets -> 选择python文件
* Add the codes below
```python

{
	// Place your snippets for python here. Each snippet is defined under a snippet name and has a prefix, body and 
	// description. The prefix is what is used to trigger the snippet and the body will be expanded and inserted. Possible variables are:
	// $1, $2 for tab stops, $0 for the final cursor position, and ${1:label}, ${2:another} for placeholders. Placeholders with the 
	// same ids are connected.
	// Example:
	// "Print to console": {
	// 	"prefix": "log",
	// 	"body": [
	// 		"console.log('$1');",
	// 		"$2"
	// 	],
	// 	"description": "Log output to console"
	// }


	
	"HEADER":{
		"prefix": "header",
		"body": [
		"#!/usr/bin/env python3",
		"# -*- encoding: utf-8 -*-",
		"'''",
		"@File    :   $TM_FILENAME",
		"@Time    :   $CURRENT_YEAR/$CURRENT_MONTH/$CURRENT_DATE $CURRENT_HOUR:$CURRENT_MINUTE:$CURRENT_SECOND",
		"@Author  :   Yan Wen ",
		"@Version :   1.0",
		"@Contact :   z19040042@s.upc.edu.cn",
		"@Desc    :   None",
		
		"'''",
		"",
		"# here put the import lib",
		"$1"
	],
	}	
}
```

