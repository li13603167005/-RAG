"""
日志工具模块
"""

import logging
import sys
from pathlib import Path
from loguru import logger
from typing import Optional


def setup_logger(
    log_file: Optional[str] = None,
    level: str = "INFO",
    format: str = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
) -> logger:
    """
    配置日志系统

    Args:
        log_file: 日志文件路径
        level: 日志级别
        format: 日志格式

    Returns:
        配置好的logger
    """
    # 移除默认处理器
    logger.remove()

    # 添加控制台处理器
    logger.add(
        sys.stderr,
        format=format,
        level=level,
        colorize=True
    )

    # 添加文件处理器
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        logger.add(
            log_file,
            format=format,
            level=level,
            rotation="10 MB",
            retention="7 days",
            compression="zip"
        )

    return logger


def get_logger(name: str = __name__) -> logger:
    """
    获取logger实例

    Args:
        name: logger名称

    Returns:
        logger实例
    """
    return logger.bind(name=name)


# 默认日志配置
default_logger = setup_logger(level="INFO")
