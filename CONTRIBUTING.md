# Contributing Guide

By contributing, you agree to the Developer Certificate of Origin
that is found in the file named `DCO` in the root directory of this
repository. All commits must include a `Signed-off-by:` line
added by running: `git commit -s -m "message"`.

## Table of Contents

- [Environment Details](#environment-details)
- [How to submit changes](#how-to-submit-changes)
- [How to report a bug](#how-to-report-a-bug)
- [How to request an "enhancement"](#how-to-request-an-enhancement)
- [Style Guide / Coding Conventions](#style-guide--coding-conventions)
- [Your First Code Contribution](#your-first-code-contribution)
- [Code of Conduct](#code-of-conduct)
- [Who is currently involved?](#who-is-currently-involved)
- [Where can I ask for help?](#where-can-i-ask-for-help)
- [Developer Certificate of Origin (DCO)](#developer-certificate-of-origin-dco)
  - [Why we chose DCO over CLA](#why-we-chose-dco-over-cla)
- [Project Roles](#project-roles)
  - [Project Lead (BDFL)](#project-lead-bdfl)
  - [Maintainer](#maintainer)
    - [Becoming a maintainer](#becoming-a-maintainer)
  - [Committer](#committer)
    - [Becoming a committer](#becoming-a-committer)
  - [Contributor](#contributor)
    - [How to contribute](#how-to-contribute)
- [List of Authors](#list-of-authors)

## Environment Details

If you are thinking about contributing to WISER with code,
then you will need to set up your environment. We have detailed
documentation on how to do this in the [Developer Environment Setup](doc/sphinx-general-wiser-docs/source/developer-content/environment-setup.md) guide.

## How to submit changes

All changes are submitted view pull requests on github. You 
will need a github account and git installed on your computer 
to get started. First you want to fork the repository so your 
github account has its own copy. Then you will clone it onto 
your computer, make some changes, push those changes to your 
fork, then make a pull request to the main WISER repository.

If that all sounded very complicated, that's okay. [This link](https://medium.com/@ravi9991ct/contributing-to-open-source-a-step-by-step-guide-to-forking-cloning-and-creating-a-pull-request-2d72dc7aeebe)
goes through the process.

For information on how your pull request will be reviewed, go 
to the [Contributing & Code Quality](doc/sphinx-general-wiser-docs/source/developer-content/contributing-and-quality.md) guide.

## How to report a bug

Read more on how to report a bug here: [Reporting a Bug](doc/sphinx-general-wiser-docs/source/contributing.md#reporting-a-bug).

## How to request an "enhancement"

Read more on how to submit a feature/enhancement request here: [Requesting a Feature](doc/sphinx-general-wiser-docs/source/contributing.md#requesting-a-feature).

## Style Guide / Coding Conventions

Many of our style and coding conventions can be found here: [Contributing & Code Quality](doc/sphinx-general-wiser-docs/source/developer-content/contributing-and-quality.md).

## Your First Code Contribution

If you are unsure where to begin contributing to WISER, you can look through our beginner or help-wanted issues.

- [Beginner Issues](https://github.com/Ehlmann-research-group/WISER/issues?q=state%3Aopen%20label%3A%22good%20first%20issue%22) - issues which
only require a few lines of code and a test or two

- [Help wanted issues](https://github.com/Ehlmann-research-group/WISER/issues?q=state%3Aopen%20label%3A%22help%20wanted%22) - More involved than a `beginner` issue, but still somewhat isolated.

## Code of Conduct

Read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for more information.

## Who is currently involved?

| Role | Name | Institution | 
|------|------|-------------|
| Project Lead [(BDFL)](https://en.wikipedia.org/wiki/Benevolent_dictator_for_life) | [Bethany Ehlmann](https://www.linkedin.com/in/bethany-ehlmann-1112b81/) | CU Boulder
| Maintainer | [Matthew Maclay](https://www.linkedin.com/in/matthew-maclay/) | LASP |
| Contributor | [Joshua Garcia-Kimble](https://www.linkedin.com/in/joshua-garcia-kimble-45211a16b/) | Caltech |

## Where can I ask for help?

You can ask for help on the [WISER discussion forum](https://github.com/Ehlmann-research-group/WISER/discussions). A 
maintainer, committer, or other contributor will be able to 
see your post and help you. Just remember, this is an 
open-source project, so it may take some time for people to 
get to your discussion post.

## Developer Certificate of Origin (DCO)

We use the [Developer Certificate of Origin (DCO)](https://github.com/Ehlmann-research-group/WISER/blob/main/DCO)
rather than a Contributor License Agreement (CLA). By signing your commits with
`git commit -s`, you automatically append a Signed-off-by: line to your commit message. This line legally certifies that you have the right to submit the code under the project's license.

### Why we chose DCO over CLA

If you don't know what a CLA or DCO is, [this link](https://www.linkedin.com/pulse/should-i-use-developers-certificate-origin-agreement-vershov-downing/)
explains it well.

We chose DCO because it keeps the barrier to contribution as low as possible.
Contributors only need to sign their commits rather than review and sign a
separate legal document. We don't want paperwork to scare away people who want
to help.

While the DCO doesn't as strongly protect WISER from contributors -- corporate
or otherwise -- who may unintentionally or intentionally violate a
corporation's copyright or patents, it does offer meaningful protection,
particularly when combined with our license.

It is worth acknowledging that "no court has ruled that tags in DVCS commit logs
can substitute for signing a contract (click-throughs, however, do constitute a
legal signature), and it's unknown if the tag is being put in place by someone
in their individual capacity or in their corporate capacity." --
[Kate Vershov Downing](https://www.linkedin.com/pulse/should-i-use-developers-certificate-origin-agreement-vershov-downing/)

Still, when coupled with our license, we are confident the DCO provides WISER with
solid protection against misuse.

## Project Roles

The WISER project has 4 roles: the project lead, maintainers, committers, and contributors. If you
have heard these terms before, great! If not, also great!
We are glad WISER is the first place you will learn about
them.

### Project Lead ([BDFL](https://en.wikipedia.org/wiki/Benevolent_dictator_for_life))
The project lead is the person who has the
final say on all major project decisions. Their role is
to lead the project in a direction that satisfies its
mission but also satisfies the community. It is not
uncommon for the Project Lead to resolve disputes on project
direction for the open-source software. The Project Lead of
WISER is Bethany Ehlmann.

### Maintainer
A maintainer doesn't have to write code. A maintainer for
WISER is defined in very broad terms: someone who has
responsibility over the direction of the project and is
committed to improving it.

#### Becoming a maintainer
Becoming a maintainer of WISER isn't a set-in-stone process.
It is mainly dependent on if the Project Lead thinks that you should
be a maintainer. To increase your odds of becoming a maintainer
it would be good to make consistent contributions to the project,
voice your opinion about the direction the project should take
in the community, and by interfacing with the Project Lead and current
maintainers. This would build trust between WISER's 'upper-brass'
and your. 

### Committer
A committer has more to do with making 'commits' to the repository.
A commit is simply a change to the repository. It is likely to either
be code or documentation.

#### Becoming a committer
Like with becoming a maintainer, becoming a committer of WISER
isn't a set-in-stone process. It is mainly dependent on if the
Project Lead and the maintainers think that you should be a committer.
To increase your odds of becoming a committer it would be good
to make active contributions involving either writing good
code or good documentation. Interfacing with the Project Lead and the
current maintainers is necessary in order to build trust.
Unlike becoming a maintainer, is not as important to actively
voice your opinion on the direction of the project. This is
because a committers primary role in the project is ensuring
quality code and documentation gets merged into the repository
as quickly as is responsible.


### Contributor
Good news, anyone can be a contributor! I love the definition
[here](https://opensource.guide/leadership-and-governance/) so much
that I will just quote it and add a bit more.

> **A “contributor” could be anyone** who comments on an
issue or pull request, people who add value to the project
(whether it’s triaging issues, writing code, or organizing
events), or anybody with a merged pull request (perhaps the
narrowest definition of a contributor).

I would like to add a contributor can also be anyone
who helps answer issues in the community or track down bugs
or actively tests the software. Really, it just means anyone
who contributes to making the WISER project better.

#### How to contribute
There are some ways to contribute that are well-defined. 
There are other ways to contribute that aren't so well-defined.

For the well-defined ways, you can can look through the issues on the
github and tackle them. Some issues will be marked with
beginner to signal that it is a good way for new contributors to
get experience with WISER. Contributions 
here can be in the form of either code or documentation or
simply a useful comment on the issue. 

If you have an idea that is not on the issue tracker, you
can make an issue for it. However, if this is a new feature
then make sure it aligns with WISER's mission and make sure
a maintainer comments on if this new feature aligns with
the mission of WISER before you code. That way you don't
put in a lot of work to not see the new feature make it into
the project. Learn more about creating issues for feature
requests here at [Requesting a Feature](doc/sphinx-general-wiser-docs/source/contributing.md#requesting-a-feature).

You can also get in contact with the maintainers or Project Lead
if you want to do other forms of contributions like triaging
issues, planning events, or anything else. A good way to 
do this is to use this github's Discussion section. Emailing
the Project Lead or maintainer is not recommended
as the forum is a better option to keep track of everything
in one place.

## List of Authors

| Role | Name | Institution | 
|------|------|-------------|
| Project Lead [(BDFL)](https://en.wikipedia.org/wiki/Benevolent_dictator_for_life) | [Bethany Ehlmann](https://www.linkedin.com/in/bethany-ehlmann-1112b81/) | CU Boulder
| Maintainer | [Matthew Maclay](https://www.linkedin.com/in/matthew-maclay/) | LASP |
| Past Maintainer | [Joshua Garcia-Kimble](https://www.linkedin.com/in/joshua-garcia-kimble-45211a16b/) | Caltech |
| Past Maintainer | [Donnie Pinkston](https://www.cms.caltech.edu/people/pinkston) | Caltech |
| Past Maintainer | [Dr. Rebecca Greenberger](https://www.linkedin.com/in/rebecca-greenberger-18842482/) | Caltech (Now The Aerospace Corporation) |
| Contributor | [Dr. Andrew Annex](https://www.linkedin.com/in/andrewannex/) | SETI Institute |
| Contributor | [Daphne Nea](https://www.linkedin.com/in/daphne-nea/) | UCLA '27 |
| Contributor | Amy Wang | Cornell '23 |
| Contributor | Sahil Azad | Caltech '25 |
