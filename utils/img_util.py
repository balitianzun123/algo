"""图片处理工具。"""

import base64
from pathlib import Path


class ImgUtil:
    """图片转换工具。"""

    @staticmethod
    def image_to_base64(
        image_path: str | Path,
    ) -> str:
        """读取图片并转换为Base64字符串。"""

        path = Path(image_path)

        if not path.exists():
            raise FileNotFoundError(
                f"图片文件不存在: {path}"
            )

        if not path.is_file():
            raise ValueError(
                f"图片路径不是文件: {path}"
            )

        with path.open("rb") as image_file:
            image_bytes = image_file.read()

        return base64.b64encode(
            image_bytes
        ).decode("utf-8")