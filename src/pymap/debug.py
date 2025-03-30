"""Debug utils."""

# Context manager that uses cProfile to profile a block of code.
import contextlib  # Importing AbstractContextManager
import cProfile
import functools
import io
import pstats
from typing import Any, Callable, Optional, Type


class ProfileBlock(contextlib.ContextDecorator):
    """Context manager for profiling a block of code."""

    def __init__(
        self,
        name: str | None = None,
        print_stats: int | None = -1,
        sort_by: str = 'cumulative',
        output_profile_file: str | None = None,
    ):
        """Initialize the profiler.

        Args:
            name (str | None): Name of the block for profiling.
            print_stats (int | None): Number of stats to print. -1 for all.
                Default is -1.
            sort_by (str): Sorting method for the stats.
            output_profile_file (str | None): File to save the profile data.
        """
        self.name = name
        self.print_stats = print_stats
        self.sort_by = sort_by
        self.output_profile_file = output_profile_file
        self.pr = cProfile.Profile()

    def __enter__(self):
        """Start profiling."""
        self.pr.enable()
        return self  # Optional

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[Any],
    ) -> Optional[bool]:
        """Stop profiling."""
        self.pr.disable()
        s = io.StringIO()
        # Print stats
        ps = pstats.Stats(self.pr, stream=s).sort_stats(self.sort_by)

        match self.print_stats:
            case -1:
                # Print all stats
                ps.print_stats()
            case 0 | None:
                # Print no stats
                pass
            case _:
                # Print the top N stats
                ps.print_stats(self.print_stats)

        # Explicitly print the stats
        print(f'Profile for {self.name or "Unnamed"}:\n{s.getvalue()}')

        if self.output_profile_file:
            self.pr.dump_stats(self.output_profile_file)

    def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]:
        """Enable use as a decorator.

        If no name is given, the function name is used.
        """
        if self.name is None:
            self.name = func.__name__

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with self:
                return func(*args, **kwargs)

        return wrapper
