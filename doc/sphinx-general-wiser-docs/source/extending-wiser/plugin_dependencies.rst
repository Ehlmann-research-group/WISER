Plugin Dependencies
===================

When making a WISER plugin, you should look to see if any of the dependencies you use match the dependencies for 
the version and platform you're on. You should pin the dependencies in your plugin to use these dependencies. If you
do not, your plugin could still work, but unexpected behavior may occur. Below you can find the dependencies for each
version of WISER and on each platform. 'Ctrl + F' / 'Cmd + F' is highly recommended to look for packages here.

.. raw:: html

    <style>
      /* Collapsible release card styling */
      .release {
        border: 3px solid #000;
        border-radius: 2px;
        margin: 12px 0;
        background: #fff;
      }
      .release > summary {
        list-style: none; /* hide default marker */
        cursor: pointer;
        padding: 14px 16px;
        font-weight: 700;
        font-size: 1.05rem;
        display: flex;
        align-items: center;
      }
      /* Custom caret */
      .release > summary::before {
        content: '\25B6'; /* triangle right */
        display: inline-block;
        margin-right: 10px;
        transform: translateY(1px);
      }
      .release[open] > summary::before {
        content: '\25BC'; /* triangle down */
      }
      /* Divider bar between stacked items */
      .release + .release {
        border-top: 3px solid #000;
      }
      .release > div {
        border-top: 3px solid #000;
        padding: 12px 16px;
        background: #fafafa;
      }
      .release p {
        margin: 0.3em 0;
      }
      .release a {
        text-decoration: underline;
      }
    </style>

.. raw:: html

    <details class="release">
      <summary>Release 1.3b1 Dependencies</summary>
      <div>

Windows dependencies: :download:`downloadable text file <resources/rel-1.3b1/windows-dependencies.txt>`

ARM Mac dependencies: :download:`downloadable text file <resources/rel-1.3b1/arm-mac-dependencies.txt>`

Intel Mac dependencies: :download:`downloadable text file <resources/rel-1.3b1/intel-mac-dependencies.txt>`

.. raw:: html

      </div>
    </details>
