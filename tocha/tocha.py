from autograd.tensor import Tensor, Arrayable, Dependency
from typing import Optional, List


class tensor(Tensor):
    def __init__(
        self,
        data: Arrayable,
        requires_grad: bool = False,
        depends_on: Optional[List[Dependency]] = None,
    ) -> None:
        super().__init__(data, requires_grad, depends_on)
