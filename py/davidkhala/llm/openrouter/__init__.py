from typing import Iterator, Protocol, Callable, List


class Result[ResultType](Protocol):
    data: List[ResultType]


class PageResponse[ResultType](Protocol):
    next: Callable[[], PageResponse | None]
    result: Result[ResultType]


def yield_list[ResultType](_: PageResponse[ResultType] | None) -> Iterator[ResultType]:
    while _ is not None:
        for item in _.result.data or []:
            yield item

        _ = _.next()
