/** Just enough DOM to render the viewer outside a browser.
 *
 * The viewer builds real DOM nodes, so the only thing standing between its
 * code and a test is `document`. This is the smallest shim that lets every
 * render function run for real -- not a mock of the views, the views
 * themselves, against payloads a live `/v1` returned.
 *
 * It is deliberately not a DOM implementation. It implements what the viewer
 * uses and throws on nothing else, so a view that starts using something new
 * fails loudly here rather than silently in a browser.
 */

class ClassList {
  constructor(element) {
    this.element = element;
  }

  get set() {
    return new Set(this.element.className.split(/\s+/).filter(Boolean));
  }

  add(name) {
    const set = this.set;
    set.add(name);
    this.element.className = [...set].join(' ');
  }

  remove(name) {
    const set = this.set;
    set.delete(name);
    this.element.className = [...set].join(' ');
  }

  contains(name) {
    return this.set.has(name);
  }

  toggle(name) {
    if (this.contains(name)) this.remove(name);
    else this.add(name);
    return this.contains(name);
  }
}

class Node {
  constructor(tag) {
    this.tagName = String(tag || '').toUpperCase();
    this.children = [];
    this.attributes = {};
    this.style = {};
    this.dataset = {};
    this.className = '';
    this.listeners = {};
    this.parentNode = null;
    this._text = '';
    this.classList = new ClassList(this);
  }

  appendChild(child) {
    child.parentNode = this;
    this.children.push(child);
    return child;
  }

  replaceChildren(...nodes) {
    this.children = [];
    for (const node of nodes) this.appendChild(node);
  }

  remove() {
    if (!this.parentNode) return;
    this.parentNode.children = this.parentNode.children.filter((child) => child !== this);
  }

  setAttribute(name, value) {
    this.attributes[name] = String(value);
  }

  getAttribute(name) {
    return this.attributes[name];
  }

  addEventListener(name, handler) {
    (this.listeners[name] ||= []).push(handler);
  }

  getContext() {
    // Canvas: the ramp draws into it and reads nothing back, so the methods
    // it calls only have to exist. `clearRect` is one of them -- the ramp
    // repaints on every hover rather than drawing over itself.
    return {
      fillStyle: '',
      fillRect() {},
      clearRect() {},
    };
  }

  getBoundingClientRect() {
    // Enough for the ramp's hover arithmetic; no test depends on the values.
    return { left: 0, top: 0, width: 100, height: 10 };
  }

  set textContent(value) {
    this._text = String(value);
    this.children = [];
  }

  get textContent() {
    if (this.children.length === 0) return this._text;
    return this.children.map((child) => child.textContent).join('');
  }

  set innerHTML(value) {
    this._text = String(value);
  }

  /** Every descendant, for assertions. */
  *walk() {
    yield this;
    for (const child of this.children) yield* child.walk();
  }

  find(predicate) {
    for (const node of this.walk()) if (predicate(node)) return node;
    return null;
  }

  findAll(predicate) {
    return [...this.walk()].filter(predicate);
  }

  hasClass(name) {
    return this.classList.contains(name);
  }
}

class TextNode extends Node {
  constructor(value) {
    super('#text');
    this._text = String(value);
  }

  get textContent() {
    return this._text;
  }
}

export function install() {
  const registry = new Map();
  const document = {
    createElement: (tag) => new Node(tag),
    createTextNode: (value) => new TextNode(value),
    createDocumentFragment: () => new Node('#fragment'),
    documentElement: new Node('html'),
    body: new Node('body'),
    addEventListener() {},
    // Stable by id, so a view that mounts into `#main` and a test that reads
    // `#main` are talking about the same node.
    getElementById: (id) => {
      if (!registry.has(id)) registry.set(id, new Node('div'));
      return registry.get(id);
    },
    execCommand() {},
  };
  globalThis.Node = Node;
  globalThis.document = document;
  globalThis.window = { addEventListener() {}, innerWidth: 1600 };
  // Node 22 defines `navigator` as a getter-only global, so it is replaced
  // rather than assigned.
  Object.defineProperty(globalThis, 'navigator', {
    value: { clipboard: { writeText: async () => {} } },
    configurable: true,
    writable: true,
  });
  globalThis.requestAnimationFrame = (callback) => callback();
  globalThis.localStorage = {
    getItem: () => null,
    setItem() {},
  };
  return { document, Node, byId: (id) => registry.get(id) };
}

export const el = (tag = 'div') => new Node(tag);
