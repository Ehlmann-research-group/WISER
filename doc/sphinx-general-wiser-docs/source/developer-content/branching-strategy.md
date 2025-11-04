# Branching Strategy (CI/CD)

WISER will be using a trunk based branching strategy with continuous integration. Release branches will be made to strictly enforce feature freezes. More explanation below.

A trunk based branching strategy means that we have our main branch and when
you want to create a feature, you will open up a feature branch from main and
make your changes in that feature branch. You should be constantly pulling from
main to ensure your feature branch is up-to-date. This next part is very **IMPORTANT**.
[Continuous integration](https://martinfowler.com/articles/branching-patterns.html#continuous-integration)
means that you should be integrating your changes into `main` very frequently.
We want very small commits to be made into main as this makes it easier to review
changes. On very rare occasions should your commit exceed 250 lines of code. As
the article linked above says, you will 'need to get used to the idea of reaching
frequent integration points with a partially built feature'. You will also have
to find a way to make sure your feature is gated from production until it is ready.
We use feature flags to do this. 

We should have good confidence that the code we merge into main is not buggy.
To have this confidence we need to ensure our testing
suite has good coverage and handles edge cases well. Currently, I would not say
WISER's testing suite has good coverage. This is due to the fact that the core body
of WISER was not written with GUI tests and the start compounded with how hard it is
to write GUI tests for PySide2 applications. There is no good GUI testing suite for
PySide2 applications currently.

Next we have release branches. Release branches will be made from main. When
release branches are made, there should not longer be any new features added
to them. All future commits on the release branch should be for fixing bugs.
After each commit is made on the release branch, you will have to merge the
release branch into `main` so main has that fix.
