(function () {
    var ALLOWED = ['ArrowLeft', 'ArrowRight', 'Enter', 'r', 's', 'a', 'b', 't', '1', '2', '3', '4', '5'];

    function isTextEntryTarget(target) {
        if (!target) return false;
        if (target.isContentEditable) return true;

        var tag = target.tagName;
        if (tag === 'TEXTAREA' || tag === 'SELECT') return true;
        if (tag !== 'INPUT') return false;

        var type = (target.getAttribute('type') || '').toLowerCase();
        return ['text', 'search', 'email', 'password', 'url', 'tel', 'number'].includes(type);
    }

    function sendKey(key) {
        var input = document.getElementById('keypress-value');
        if (!input) return;
        var setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
        setter.call(input, key + ':' + Date.now());
        input.dispatchEvent(new Event('input', { bubbles: true }));
    }

    document.addEventListener('keydown', function (e) {
        if (isTextEntryTarget(e.target)) return;
        if (!ALLOWED.includes(e.key)) return;
        e.preventDefault();
        sendKey(e.key);
    });
})();
