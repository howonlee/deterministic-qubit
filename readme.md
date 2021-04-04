On the Deterministic Qubit
===

The deterministic qubit is the propositional variable. Peep the writeup in writeup.pdf for more.

`int_trans.py` has a test of the propositional Fourier transform. `kron.py` has a test of the entanglement. You can run them straight off or look at the code.

The claimed speedups for entangled states and integral transforms happen empirically, too. Here they are. This is median of 5 runs: machine effects seem to futz up the proper shape of the asymptotics of the propositional stuff because the times are too fast. Run `timing.py` to replicate these or futz with parameters.

![](https://i.imgur.com/9xrQDVH.png)
![](https://i.imgur.com/HzePhwt.png)

AAQ (Actually Asked Questions)
----

Have you considered for a second that threats to encryption for everyone might not be a good thing to do in current circumstances?
---
It's randoms like me publishing or nation-states figuring it out and actually using it for exploits.

Also, I'm pretty sure this also might be possible to leverage for the original proposed use of quantum computation (quantum chemistry simulation), which might be of strong value in the current world crisis, although I actually don't know anything about quantum chemistry simulation.

"I think I have a thing arguably worth many many millions even without generalizability, but screw it, I'm just so excited I'll tell the world right now!"
----
The only folks who have a strong ability to pay are intelligence agencies and people who are adjacent to intelligence agencies (the set of people who employ cryptographers outside of academia). Or organized crime.

I am scared to death of any and every intelligence agency _and_ organized crime because any right-thinking person should be, so I can't really reasonably get paid.

Also if the Grover's thing works out I don't know about money still existing. It probably won't work out, though.

Should I go and freak out now?
---
Probably not at least for a few weeks. I don't know about after that.

How does this handle entangled states? Product states aren't a significant volume of the whole space
----
See section 4.1 of the writeup.

Why don't you do a delayed ("responsible") disclosure?
---
This is the best I could do, given the massive size of the set of stakeholders, sorry.

Why did you do this?
---
For the comedy value, to be honest.
