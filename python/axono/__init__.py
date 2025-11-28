"""
Axono - High Performance Computing Library
"""

from .core import *

__version__ = "0.1.0"
__author__ = "ByteRainLab"
__description__ = "High performance computing library for big data processing"


def welcome():
    text = '''                                                                                                                                                   
       db         8b        d8  ,ad8888ba,    888b      88    ,ad8888ba,                                              88                           
      d88b         Y8,    ,8P  d8"'    `"8b   8888b     88   d8"'    `"8b                                             ""                           
     d8'`8b         `8b  d8'  d8'        `8b  88 `8b    88  d8'        `8b                                                                         
    d8'  `8b          Y88P    88          88  88  `8b   88  88          88       ,adPPYba,  8b,dPPYba,    ,adPPYb,d8  88  8b,dPPYba,    ,adPPYba,  
   d8YaaaaY8b         d88b    88          88  88   `8b  88  88          88      a8P_____88  88P'   `"8a  a8"    `Y88  88  88P'   `"8a  a8P_____88  
  d8""""""""8b      ,8P  Y8,  Y8,        ,8P  88    `8b 88  Y8,        ,8P      8PP"""""""  88       88  8b       88  88  88       88  8PP"""""""  
 d8'        `8b    d8'    `8b  Y8a.    .a8P   88     `8888   Y8a.    .a8P       "8b,   ,aa  88       88  "8a,   ,d88  88  88       88  "8b,   ,aa  
d8'          `8b  8P        Y8  `"Y8888Y"'    88      `888    `"Y8888Y"'         `"Ybbd8"'  88       88   `"YbbdP"Y8  88  88       88   `"Ybbd8"'  
                                                                                                          aa,    ,88                               
                                                                                                           "Y8bbdP"                                
Dear 使用者

引擎版本: {__version__}
欢迎使用Axono Ai 引擎~

Axono的官方团队为您送上诚挚的问候!

Best regards,
{__author__}'''
    text = text.replace("{__version__}", __version__)
    text = text.replace("{__author__}", __author__)
    print(text)
